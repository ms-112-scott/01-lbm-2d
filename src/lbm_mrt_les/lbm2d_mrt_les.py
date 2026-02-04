import matplotlib
import numpy as np
from matplotlib import cm
import taichi as ti
import taichi.math as tm

from VideoRecorder import VideoRecorder
from scipy.ndimage import gaussian_filter


import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import utils

ti.init(arch=ti.gpu)


@ti.data_oriented
class LBM2D_MRT_LES:
    # ------------------------------------------------
    # region init
    def __init__(self, config, mask_data=None):
        """
        LBM Solver 初始化入口
        """
        self.config = config

        # 1. 讀取模擬與物理參數
        self._init_params()

        # 2. 配置 Taichi 記憶體 (Fields)
        self._init_fields(mask_data)

        # 3. 定義 LBM/MRT 常數與矩陣 (優化重點)
        self._init_constants()

        # # 4. 初始化場數值 (由 0 開始或讀取設定)
        # self.init_sim()

    #  init 子函式: 參數讀取
    def _init_params(self):
        sim_cfg = self.config["simulation"]

        # 基礎幾何與時間
        self.name = sim_cfg["name"]
        self.nx = sim_cfg["nx"]
        self.ny = sim_cfg["ny"]
        self.steps_per_frame = sim_cfg.get("steps_per_frame", 10)
        self.warmup_steps = sim_cfg.get("warmup_steps", 0)

        # 物理參數
        self.niu = sim_cfg["niu"]
        self.tau_0 = 3.0 * self.niu + 0.5

        # LES (大渦模擬) 參數
        self.C_smag = sim_cfg.get("smagorinsky_constant", 0.15)
        self.Cs_sq_factor = 18.0 * (self.C_smag**2)

        # MRT 鬆弛參數
        self.S_other = sim_cfg.get("ghost_moments_s", 1.2)

        # 視覺化參數
        self.viz_sigma = sim_cfg.get("visualization_gaussian_sigma", 1.0)

    #  init 子函式: 記憶體配置 (Fields)
    def _init_fields(self, mask_data):
        # 巨觀量 (Macro-scopic)
        self.rho = ti.field(dtype=ti.f32, shape=(self.nx, self.ny))
        self.vel = ti.Vector.field(2, dtype=ti.f32, shape=(self.nx, self.ny))

        # 分布函數 (Micro-scopic, f_old / f_new)
        self.f_old = ti.Vector.field(9, dtype=ti.f32, shape=(self.nx, self.ny))
        self.f_new = ti.Vector.field(9, dtype=ti.f32, shape=(self.nx, self.ny))

        # 遮罩 (Mask)
        self.mask = ti.field(dtype=ti.f32, shape=(self.nx, self.ny))
        if mask_data is not None:
            self.mask.from_numpy(mask_data.astype(np.float32))
        else:
            self.mask.fill(0.0)

        # 邊界條件 (Boundary Conditions)
        bc_cfg = self.config["boundary_condition"]
        self.bc_type = ti.field(dtype=ti.i32, shape=4)
        self.bc_value = ti.Vector.field(2, dtype=ti.f32, shape=4)

        self.bc_type.from_numpy(np.array(bc_cfg["type"], dtype=np.int32))
        self.bc_value.from_numpy(np.array(bc_cfg["value"], dtype=np.float32))

        # 統計與計數
        self.frame_count = ti.field(dtype=ti.i32, shape=())

    #  init 子函式: 常數與矩陣 (Constants)
    def _init_constants(self):
        # D2Q9 權重
        self.w = ti.types.vector(9, ti.f32)(
            4.0 / 9.0,
            1.0 / 9.0,
            1.0 / 9.0,
            1.0 / 9.0,
            1.0 / 9.0,
            1.0 / 36.0,
            1.0 / 36.0,
            1.0 / 36.0,
            1.0 / 36.0,
        )

        # D2Q9 離散速度向量
        self.e = ti.types.matrix(9, 2, ti.i32)(
            [
                [0, 0],
                [1, 0],
                [0, 1],
                [-1, 0],
                [0, -1],
                [1, 1],
                [-1, 1],
                [-1, -1],
                [1, -1],
            ]
        )

        # --- MRT 轉換矩陣 (核心優化) ---
        # 這裡將 M 定義為 Taichi Matrix 而非 Field，
        # 定義 9x9 的場來儲存矩陣
        self.M_field = ti.field(dtype=ti.f32, shape=(9, 9))
        self.invM_field = ti.field(dtype=ti.f32, shape=(9, 9))

        # 定義 numpy 數據
        M_np = np.array(
            [
                [1, 1, 1, 1, 1, 1, 1, 1, 1],
                [-4, -1, -1, -1, -1, 2, 2, 2, 2],
                [4, -2, -2, -2, -2, 1, 1, 1, 1],
                [0, 1, 0, -1, 0, 1, -1, -1, 1],
                [0, -2, 0, 2, 0, 1, -1, -1, 1],
                [0, 0, 1, 0, -1, 1, 1, -1, -1],
                [0, 0, -2, 0, 2, 1, 1, -1, -1],
                [0, 1, -1, 1, -1, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 1, -1, 1, -1],
            ],
            dtype=np.float32,
        )

        invM_np = np.linalg.inv(M_np).astype(np.float32)

        # 將數據寫入 Field (這一步會在 Python 端執行一次)
        self.M_field.from_numpy(M_np)
        self.invM_field.from_numpy(invM_np)

        # MRT 鬆弛對角矩陣 (S vector)
        # 用於碰撞步驟: m* = m - S * (m - m_eq)
        # S_other 是 Ghost Moments 的鬆弛率
        self.S_base = ti.types.vector(9, ti.f32)(
            0.0,  # density (conserved)
            self.S_other,  # energy
            self.S_other,  # epsilon
            0.0,  # jx (conserved)
            self.S_other,  # qx
            0.0,  # jy (conserved)
            self.S_other,  # qy
            0.0,  # pxx (conserved in standard LBM, but relaxed here)
            0.0,  # pxy
        )

    # endregion

    # ------------------------------------------------
    # region LBM MRT-LES Kernels and Functions
    def get_physical_fields(self):
        """
        將 GPU 上的速度場與遮罩數據導出為 NumPy 數組
        """
        # .to_numpy() 會自動處理數據同步與拷貝
        return self.vel.to_numpy(), self.mask.to_numpy()

    @ti.func
    def f_eq(self, i, j):
        eu = self.e @ self.vel[i, j]
        uv = tm.dot(self.vel[i, j], self.vel[i, j])
        return self.w * self.rho[i, j] * (1 + 3 * eu + 4.5 * eu * eu - 1.5 * uv)

    @ti.func
    def get_meq(self, rho, u, v):
        u2 = u * u + v * v
        return ti.types.vector(9, float)(
            rho,
            rho * (-2.0 + 3.0 * u2),
            rho * (1.0 - 3.0 * u2),
            rho * u,
            -rho * u,
            rho * v,
            -rho * v,
            rho * (u * u - v * v),
            rho * u * v,
        )

    @ti.kernel
    def init(self):
        self.vel.fill(0)
        self.rho.fill(1)
        self.frame_count[None] = 0
        for i, j in self.rho:
            self.f_old[i, j] = self.f_new[i, j] = self.f_eq(i, j)

    @ti.kernel
    def collide_and_stream(self):
        for i, j in ti.ndrange((1, self.nx - 1), (1, self.ny - 1)):
            f_temp = ti.types.vector(9, float)(0.0)
            for k in ti.static(range(9)):
                ip, jp = i - self.e[k, 0], j - self.e[k, 1]
                f_temp[k] = self.f_old[ip, jp][k]

            m = ti.types.vector(9, float)(0.0)
            for r in ti.static(range(9)):
                val = 0.0
                for c in ti.static(range(9)):
                    val += self.M_field[r, c] * f_temp[c]
                m[r] = val

            rho_l = m[0]
            u_l, v_l = 0.0, 0.0
            if rho_l > 0:
                u_l, v_l = m[3] / rho_l, m[5] / rho_l

            m_eq = self.get_meq(rho_l, u_l, v_l)
            neq_7 = m[7] - m_eq[7]
            neq_8 = m[8] - m_eq[8]
            momentum_neq_mag = tm.sqrt(neq_7 * neq_7 + neq_8 * neq_8)

            # 1. 基礎 LES 計算 (保持原樣)
            tau_eff = self.tau_0
            if self.C_smag > 0.001:
                term_inside = (
                    self.tau_0**2 + (self.Cs_sq_factor * momentum_neq_mag) / rho_l
                )
                tau_eddy = 0.5 * (tm.sqrt(term_inside) - self.tau_0)
                tau_eff = self.tau_0 + tau_eddy

            # ==========================================
            # [新增] 2. Sponge Layer (阻尼層)
            # ==========================================
            sponge_width = 200  # 阻尼層寬度 (Lattice Units)
            sponge_strength = 1.0  # 增加的額外 Tau 值 (越大越黏)

            # 判斷是否在右側邊界區域
            dist_from_inlet = i
            x_start_sponge = self.nx - sponge_width

            if dist_from_inlet > x_start_sponge:
                # 歸一化座標 (0.0 ~ 1.0)
                coord = (dist_from_inlet - x_start_sponge) / sponge_width
                # 使用二次曲線平滑增加黏滯性
                tau_eff += sponge_strength * (coord * coord)
            # ==========================================

            s_eff = 1.0 / tau_eff
            S_local = self.S_base

            # MRT 的 S7, S8 控制剪切黏滯性，必須使用 s_eff
            S_local[7] = s_eff
            S_local[8] = s_eff

            m_star = m - S_local * (m - m_eq)

            f_new_val = ti.types.vector(9, float)(0.0)
            for r in ti.static(range(9)):
                val = 0.0
                for c in ti.static(range(9)):
                    val += self.invM_field[r, c] * m_star[c]
                f_new_val[r] = val

            self.f_new[i, j] = f_new_val

    @ti.kernel
    def update_macro_var(self):
        for i, j in ti.ndrange((1, self.nx - 1), (1, self.ny - 1)):
            local_rho = 0.0
            local_vel = tm.vec2(0.0, 0.0)
            for k in ti.static(range(9)):
                self.f_old[i, j][k] = self.f_new[i, j][k]
                local_rho += self.f_new[i, j][k]
                local_vel += tm.vec2(self.e[k, 0], self.e[k, 1]) * self.f_new[i, j][k]

            self.rho[i, j] = local_rho
            if local_rho > 0:
                self.vel[i, j] = local_vel / local_rho
            else:
                self.vel[i, j] = tm.vec2(0, 0)

    @ti.kernel
    def apply_bc(self):
        self.frame_count[None] += 1
        # 緩啟動
        ramp = tm.min(1.0, float(self.frame_count[None]) / self.warmup_steps)

        for j in range(1, self.ny - 1):
            self.apply_bc_core(1, 0, 0, j, 1, j, ramp)
            self.apply_bc_core(1, 2, self.nx - 1, j, self.nx - 2, j, ramp)
        for i in range(self.nx):
            self.apply_bc_core(1, 1, i, self.ny - 1, i, self.ny - 2, ramp)
            self.apply_bc_core(1, 3, i, 0, i, 1, ramp)

        for i, j in ti.ndrange(self.nx, self.ny):
            if self.mask[i, j] == 1.0:
                self.vel[i, j] = 0.0, 0.0
                self.f_old[i, j] = self.f_eq(i, j)

    @ti.func
    def apply_bc_core(self, outer, dr, ibc, jbc, inb, jnb, ramp: float):
        if outer == 1:
            if self.bc_type[dr] == 0:
                self.vel[ibc, jbc] = self.bc_value[dr] * ramp
            elif self.bc_type[dr] == 1:
                self.vel[ibc, jbc] = self.vel[inb, jnb]
        self.rho[ibc, jbc] = self.rho[inb, jnb]
        self.f_old[ibc, jbc] = (
            self.f_eq(ibc, jbc) - self.f_eq(inb, jnb) + self.f_old[inb, jnb]
        )

    def check_re(self):
        u_vec = self.bc_value[0]
        u_char = np.sqrt(u_vec[0] ** 2 + u_vec[1] ** 2)
        # 嘗試從 config 讀取特徵長度 (CL), 若無則用預設值
        l_char = self.config["boundary_condition"].get("CL", 20.0)
        if self.config["mask"]["type"] == "cylinder":
            l_char = self.config["mask"]["params"]["r"] * 2

        print(f"--- [LES Info] ---")
        print(f"Smagorinsky Constant (Cs): {self.C_smag}")
        print(f"Ghost Moments S: {self.S_other} (Read from Config)")
        utils.print_reynolds_info(u_char, l_char, self.niu, "Characteristic Length")
        return (u_char * l_char) / self.niu

    def run_step(self, steps=1):
        """
        [核心運算層]
        執行 LBM 的物理時間步推進。

        標準循環：
        1. Collide & Stream: 計算分佈函數的碰撞與流動 (f_old -> f_new)
        2. Update Macro: 計算密度與速度，並交換緩衝區 (f_new -> f_old)
        3. Apply BC: 強制設定邊界條件與障礙物處理
        """
        for _ in range(steps):
            # 1. 碰撞與串流 (計算 f_new)
            self.collide_and_stream()

            # 2. 更新巨觀量 (rho, vel) 並將 f_new 寫回 f_old
            # 注意：你的 update_macro_var kernel 內包含了 f_old[i,j] = f_new[i,j]
            # 這一步至關重要，否則下一幀計算會用到舊數據
            self.update_macro_var()

            # 3. 應用邊界條件 (Inlet/Outlet) 與 障礙物 (Mask)
            # 這會覆蓋邊界上的 vel 和 rho，確保物理場正確
            self.apply_bc()

    # endregion
    # ------------------------------------------------
