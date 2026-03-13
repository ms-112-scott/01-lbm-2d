"""
config_utils/context.py

定義在函式間流動的 dict 結構（取代長引數串）。

設計原則：
  - 所有 dict 都是普通 Python dict，不使用 TypedDict 強制型別
  - 每個 context 的 key 集合在本檔案的 docstring 中明確列出
  - 函式只需 import 用到的 key，不需要整包傳遞

【SimContext】  全域模擬設定，從 master_config 解析一次，整個批次共享
  keys:
    rho_in          float  基準入口密度（高阻塞時各 case 自動降低）
    rho_out         float  出口密度
    nu_lb_list      list   nu 候選清單
    warmup_passes   float  暖機 CTU 倍數
    total_passes    float  總模擬 CTU 倍數
    start_record_passes float  開始錄製 CTU 倍數
    saves_per_ctu   float  每 CTU 儲存幀數
    c_smag          float  Smagorinsky 常數
    U_phys          float  物理入口風速（m/s，取清單第一個）
    nu_air          float  空氣動黏度（m²/s）
    blockage_buffer int    阻塞率計算時的右側緩衝（px）
    mask_invert     bool   PNG 固體判斷是否反轉
    project_name    str
    data_save_root  str    輸出資料根目錄
    output_dir      str    YAML 輸出目錄
    base_template   dict   master_config 的 template 區塊
    physical_constants dict master_config 的 physical_constants 區塊

【MaskContext】  單一 mask 的幾何資訊，從 metadata.json + mask PNG 計算
  keys:
    mask_path   str    PNG 檔案完整路徑
    mask_stem   str    檔名不含副檔名（e.g. "mask_01"）
    nx          int    domain X 格點數（來自 metadata）
    ny          int    domain Y 格點數（來自 metadata）
    pad_right   int    右側 sponge padding（px）
    pad_top     int    上方 padding（px）
    pad_bot     int    下方 padding（px）
    pad_left    int    左方 padding（px）
    l_char      int    最大單一建築特徵長度（px）
    max_blockage float 最嚴重截面阻塞率（0~1）

【CaseResult】  單一 case 的計算結果，在 process_mask 內部各步驟間流動
  keys:
    rho_in_case    float  調整後的入口密度
    u_inlet_safe   float  對應的安全入口速度
    open_fraction  float  有效開放比
    nu_lb          float  最終選用的 nu 值
    nu_re_pairs    list   [(nu, Re), ...] 可行清單
    u_bernoulli    float  Bernoulli 速度估算
    Ma             float  馬赫數
    Re             float  Reynolds 數
    tau            float  鬆弛時間
    dx_mm          float  物理格點間距（mm，僅供顯示）
    steps_per_ctu  int
    warmup_steps   int
    max_steps      int
    start_record_step int
    interval       int
    config_filename str   輸出 YAML 檔名
    sim_name        str   模擬識別名稱
"""


def build_sim_context(master_cfg: dict) -> dict:
    """
    從 master_config dict 建構 SimContext。

    同時處理 nu_lb_list fallback 與 U_phys list 取第一個值的邊界情況。
    """
    settings = master_cfg["settings"]
    physics = master_cfg["physics_control"]
    phys_const = master_cfg["physical_constants"]
    base_template = master_cfg["template"]

    project_name = settings["project_name"]
    project_dir = f"SimCases/{project_name}"

    nu_lb_list = physics.get("nu_lb_list")
    if not nu_lb_list:
        nu_single = physics["nu"]
        nu_lb_list = [nu_single]
        print(f"[Info] 未找到 nu_lb_list，使用單一 nu={nu_single}。")

    U_phys_raw = phys_const["inlet_velocity_ms"]
    U_phys = U_phys_raw[0] if isinstance(U_phys_raw, list) else U_phys_raw

    return {
        # 壓力邊界
        "rho_in": physics["rho_in"],
        "rho_out": physics["rho_out"],
        # nu 候選
        "nu_lb_list": nu_lb_list,
        # 步數倍率
        "warmup_passes": physics["warmup_passes"],
        "total_passes": physics["total_passes"],
        "start_record_passes": physics["start_record_passes"],
        "saves_per_ctu": physics["saves_per_physical_second"],
        # 模型參數
        "c_smag": physics["smagorinsky_constant"],
        # 物理換算
        "U_phys": U_phys,
        "nu_air": phys_const.get("kinematic_viscosity_air_m2_s", 1.5e-5),
        # 幾何設定
        "blockage_buffer": settings.get("blockage_buffer", 128),
        "mask_invert": base_template.get("mask", {}).get("invert", False),
        # 路徑
        "project_name": project_name,
        "data_save_root": f"outputs/{project_name}",
        "output_dir": f"{project_dir}/configs",
        "mask_dir": f"{project_dir}/masks",
        "mask_meta_dir": f"{project_dir}",
        # 模板資料（組裝 YAML 用）
        "base_template": base_template,
        "physical_constants": master_cfg["physical_constants"],
    }


def build_mask_context(mask_path: str, meta_entry: dict) -> dict:
    """
    從 mask 路徑與 metadata.json 的單筆 entry 建構 MaskContext。

    Args:
        mask_path  : PNG 完整路徑
        meta_entry : metadata.json 中對應這個 mask 的 dict
                     必須包含 domain_W_total, domain_H_total, pad_right,
                     pad_top, pad_bot, pad_left
    """
    import os

    mask_stem = os.path.splitext(os.path.basename(mask_path))[0]
    return {
        "mask_path": mask_path,
        "mask_stem": mask_stem,
        "nx": int(meta_entry["domain_W_total"]),
        "ny": int(meta_entry["domain_H_total"]),
        "pad_right": int(meta_entry["pad_right"]),
        "pad_top": int(meta_entry["pad_top"]),
        "pad_bot": int(meta_entry["pad_bot"]),
        "pad_left": int(meta_entry["pad_left"]),
        # l_char / max_blockage 由 geometry 計算後填入
        "l_char": None,
        "max_blockage": None,
    }
