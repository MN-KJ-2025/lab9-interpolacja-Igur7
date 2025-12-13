# =================================  TESTY  ===================================
# Testy do tego pliku obejmują jedynie weryfikację poprawności wyników dla
# prawidłowych danych wejściowych - obsługa niepoprawych danych wejściowych
# nie jest ani wymagana ani sprawdzana. W razie potrzeby lub chęci można ją 
# wykonać w dowolny sposób we własnym zakresie.
# =============================================================================
import numpy as np


def chebyshev_nodes(n: int = 10) -> np.ndarray | None:
    if not isinstance(n,int) or n <= 0:
        return None
    return np.cos(np.pi * np.arange(0,n) / (n - 1))


def bar_cheb_weights(n: int = 10) -> np.ndarray | None:
    if not isinstance(n,int) or n<= 0:
        return None
    w = np.ones(n) 
    w[0] = 0.5
    w[-1] = (-1)**(n-1) * 0.5
    w[1:-1:2] = -1
    return w


def barycentric_inte(
xi: np.ndarray, yi: np.ndarray, wi: np.ndarray, x: np.ndarray
) -> np.ndarray | None:

    """Funkcja wykonująca interpolację barycentryczną.

    Args:
        xi (np.ndarray): Węzły interpolacji (n,).
        yi (np.ndarray): Wartości funkcji w węzłach (n,).
        wi (np.ndarray): Wagi barycentryczne (n,).
        x (np.ndarray): Punkty, w których ma być obliczona wartość 
            interpolowanego wielomianu (m,).

    Returns:
        np.ndarray: Wartości interpolowanego wielomianu w punktach x (m,).
        Jeżeli dane wejściowe są niepoprawne funkcja zwraca `None`.
    """
    if not all(isinstance(arr, np.ndarray) for arr in (xi, yi, wi, x)):
        return None
    if not (xi.ndim == yi.ndim == wi.ndim == 1):
        return None
    if xi.ndim != 1 or yi.ndim != 1 or wi.ndim != 1 or x.ndim != 1:
        return None
    if xi.size == 0 or x.size == 0:
        return None
    P = np.empty_like(x, dtype=float) #tablica wynikowa
    
    for j,xj in enumerate(x):
        diff = xj - xi
        idx = np.where(diff == 0)[0]
        if idx.size > 0:
            P[j] = yi[idx[0]]
            continue
        numerator = np.sum(wi * yi / diff)
        denominator = np.sum(wi / diff)
        P[j] = numerator / denominator
    return P

def L_inf(
    xr: int | float | list | np.ndarray, x: int | float | list | np.ndarray
) -> float | None:
    """Funkcja obliczająca normę L-nieskończoność. Powinna działać zarówno na 
    wartościach skalarnych, listach, jak i wektorach biblioteki numpy.

    Args:
        xr (int | float | list | np.ndarray): Wartość dokładna w postaci 
            skalara, listy lub wektora (n,).
        x (int | float | list | np.ndarray): Wartość przybliżona w postaci 
            skalara, listy lub wektora (n,).

    Returns:
        (float): Wartość normy L-nieskończoność.
        Jeżeli dane wejściowe są niepoprawne funkcja zwraca `None`.
    """
    try:
        xr_arr = np.asarray(xr)
        x_arr = np.asarray(x)
        if xr_arr.shape != x_arr.shape:
            return None
        return np.max(np.abs(xr_arr - x_arr))
    except:
        return None