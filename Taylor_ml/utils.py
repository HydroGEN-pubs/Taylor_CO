from datetime import date
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt


# All available forcing variable names - these should not need to be changed
FORCING_VARIABLES = ('DSWR', 'DLWR', 'APCP', 'Temp', 'UGRD', 'VGRD', 'Press', 'SPFH')

# All available CLM output names - these should not need to be changed
CLM_OUTPUT_VARIABLES = (
    'eflx_lh_tot',
    'eflx_lwrad_out',
    'eflx_sh_tot',
    'eflx_soil_grnd',
    'qflx_evap_tot',
    'qflx_evap_grnd',
    'qflx_evap_soi',
    'qflx_evap_veg',
    'qflx_tran_veg',
    'qflx_infl',
    'swe_out',
    't_grnd',
    'qflx_qirr',
    't_soil'
)


def hours_in_year(year, water_year=True):
    if water_year:
        yyyy, mm, dd = year-1, 10, 1
    else:
        yyyy, mm, dd = year, 1, 1
    return (date(yyyy+1, mm, dd) - date(yyyy, mm, dd)).days * 24


def compute_RMSD(y, y_pred):
    ngrid = y.size
    return np.sqrt(1 / ngrid * (np.sum( (y - y_pred)**2 )))


def compute_NSE(y, y_pred):
    return 1 - np.sum((y - y_pred)**2)/np.sum((y - np.mean(y))**2)


def compute_KGE(y, y_pred):
    r, _ = stats.pearsonr(y.flatten(), y_pred.flatten())
    a = np.std(y_pred.flatten()) / np.std(y.flatten())
    b = np.mean(y_pred.flatten()) / np.mean(y.flatten())
    return 1 - np.sqrt((r - 1)**2 + (a - 1)**2 + (b - 1)**2)


def compute_stats(y, y_pred,mask=None):
    if mask is not None:
        y = y[mask==True]
        y_pred = y_pred[mask==True]
    return np.array([compute_RMSD(y, y_pred), compute_NSE(y, y_pred), compute_KGE(y, y_pred)])
    
def plot(y_hat, y_pred, filepath):
    fig, axes = plt.subplots(1, 3)
    im = axes[0].imshow(y_hat, cmap=plt.get_cmap('hot'), interpolation='nearest')
    fig.colorbar(im, ax=axes[0])
    axes[0].set_title(f'y_hat')
    im = axes[1].imshow(y_pred, cmap=plt.get_cmap('hot'), interpolation='nearest')
    fig.colorbar(im, ax=axes[1])
    axes[1].set_title(f'y_pred')
    im = axes[2].imshow(y_hat - y_pred, cmap=plt.get_cmap('hot'), interpolation='nearest')
    fig.colorbar(im, ax=axes[2])
    axes[2].set_title(f'y_hat-y_pred')

    plt.savefig(filepath)
    plt.close(fig)
