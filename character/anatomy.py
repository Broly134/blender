"""Table anthropometrique du personnage.

Repere : origine au menton, +X vers la gauche du personnage, +Y vers l'avant
(la direction du regard), +Z vers le haut. Tout est en metres.

Les valeurs viennent des proportions classiques d'un crane masculin adulte
(hauteur menton -> vertex ~ 23 cm) ajustees d'apres la photo de reference :
visage large, machoire carree, nez large a pointe charnue, levres pleines.
"""

from .mathutil import Profile

# --- reperes verticaux -----------------------------------------------------
MENTON = 0.000          # bas du menton
SULCUS = 0.0215         # sillon mento-labial
LOWER_LIP = 0.0292      # labiale inferius
STOMION = 0.0405        # ligne de fermeture des levres
UPPER_LIP = 0.0495      # labiale superius
SUBNASALE = 0.0640      # base du nez
ALAR = 0.0700           # ailes du nez, point le plus large
NOSE_TIP = 0.0755
EYE_LINE = 0.1125       # centre pupillaire
NASION = 0.1180         # racine du nez
GLABELLA = 0.1275
BROW = 0.1330           # sommet de l'arcade sourciliere
TRICHION = 0.1820       # naissance des cheveux
VERTEX = 0.2265         # sommet du crane
GONION = 0.0460         # angle de la machoire
NECK_BASE = -0.1550

# --- reperes horizontaux ---------------------------------------------------
EYE_X = 0.0315          # demi-ecart pupillaire
EYE_R = 0.01225         # rayon du globe oculaire
EYE_Y = 0.0645          # centre du globe (profondeur)
IRIS_R = 0.00590
PUPIL_R = 0.00215
MOUTH_HALF = 0.0268     # demi-largeur de la bouche
ALAR_HALF = 0.0245      # demi-largeur des ailes du nez
EAR_X = 0.0735
EAR_Y = -0.0175
EAR_Z = 0.1010

# --- profils de la coupe horizontale ---------------------------------------
# W : demi-largeur ; F : avancee maximale (+Y) ; B : recul maximal (-Y)

HALF_WIDTH = Profile([
    (-0.1550, 0.0830),
    (-0.1250, 0.0735),
    (-0.1000, 0.0688),
    (-0.0750, 0.0658),
    (-0.0500, 0.0638),
    (-0.0280, 0.0622),
    (-0.0120, 0.0600),
    (0.0000, 0.0468),
    (0.0120, 0.0538),
    (0.0250, 0.0598),
    (0.0460, 0.0672),
    (0.0700, 0.0722),
    (0.0900, 0.0752),
    (0.1125, 0.0778),
    (0.1330, 0.0793),
    (0.1500, 0.0798),
    (0.1700, 0.0776),
    (0.1900, 0.0700),
    (0.2000, 0.0620),
    (0.2100, 0.0500),
    (0.2180, 0.0360),
    (0.2240, 0.0195),
    (0.2265, 0.0060),
])

FRONT = Profile([
    (-0.1550, 0.0480),
    (-0.1250, 0.0320),
    (-0.1000, 0.0250),
    (-0.0750, 0.0225),
    (-0.0500, 0.0235),
    (-0.0300, 0.0300),
    (-0.0180, 0.0400),
    (-0.0080, 0.0540),
    (0.0000, 0.0655),
    (0.0130, 0.0712),
    (0.0270, 0.0722),
    (0.0440, 0.0760),
    (0.0640, 0.0796),
    (0.0850, 0.0814),
    (0.1000, 0.0826),
    (0.1125, 0.0834),
    (0.1200, 0.0846),
    (0.1275, 0.0860),
    (0.1400, 0.0852),
    (0.1600, 0.0796),
    (0.1800, 0.0700),
    (0.1900, 0.0640),
    (0.2000, 0.0540),
    (0.2100, 0.0415),
    (0.2180, 0.0288),
    (0.2240, 0.0155),
    (0.2265, 0.0050),
])

BACK = Profile([
    (-0.1550, 0.0810),
    (-0.1250, 0.0740),
    (-0.1000, 0.0710),
    (-0.0750, 0.0708),
    (-0.0500, 0.0722),
    (-0.0250, 0.0752),
    (-0.0050, 0.0778),
    (0.0100, 0.0800),
    (0.0460, 0.0868),
    (0.0700, 0.0918),
    (0.0900, 0.0958),
    (0.1125, 0.1000),
    (0.1330, 0.1034),
    (0.1500, 0.1050),
    (0.1700, 0.1032),
    (0.1900, 0.0942),
    (0.2000, 0.0818),
    (0.2100, 0.0640),
    (0.2180, 0.0440),
    (0.2240, 0.0235),
    (0.2265, 0.0070),
])

EXP_FRONT = Profile([
    (-0.1550, 2.05),
    (-0.0400, 2.05),
    (-0.0100, 1.90),
    (0.0100, 1.85),
    (0.0300, 2.10),
    (0.0500, 2.45),
    (0.0750, 2.80),
    (0.1000, 3.05),
    (0.1250, 3.05),
    (0.1450, 2.80),
    (0.1700, 2.50),
    (0.2000, 2.20),
    (0.2265, 2.02),
])

EXP_BACK = Profile([
    (-0.1550, 2.10),
    (-0.0400, 2.12),
    (0.0200, 2.22),
    (0.0700, 2.28),
    (0.1200, 2.22),
    (0.1700, 2.14),
    (0.2265, 2.02),
])

# --- profil du nez ---------------------------------------------------------
# hauteur du dos du nez au-dessus du plan facial
NOSE_RIDGE = Profile([
    (0.0555, 0.0000),
    (0.0605, 0.0038),
    (0.0640, 0.0098),
    (0.0690, 0.0180),
    (0.0755, 0.0226),
    (0.0820, 0.0208),
    (0.0900, 0.0170),
    (0.0990, 0.0126),
    (0.1080, 0.0080),
    (0.1180, 0.0028),
    (0.1290, 0.0000),
])

# demi-largeur du nez
NOSE_HALF = Profile([
    (0.0545, 0.0152),
    (0.0620, 0.0200),
    (0.0700, 0.0216),
    (0.0760, 0.0202),
    (0.0830, 0.0168),
    (0.0920, 0.0136),
    (0.1030, 0.0112),
    (0.1150, 0.0100),
    (0.1290, 0.0106),
])
