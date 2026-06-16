from dataclasses import dataclass


@dataclass(frozen=True)
class MaskVCTGuidance:
    omega_all: float
    omega_spk: float
    omega_ling: float


MASKVCT_ALL = MaskVCTGuidance(omega_all=1.5, omega_spk=1.0, omega_ling=1.0)
MASKVCT_SPK = MaskVCTGuidance(omega_all=0.0, omega_spk=2.0, omega_ling=0.5)
MASKVCT_SPK_ACCENT = MaskVCTGuidance(omega_all=0.0, omega_spk=2.5, omega_ling=0.5)

