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

    if not (isinstance(xi, np.ndarray) and isinstance(yi, np.ndarray) and isinstance(wi, np.ndarray) and isinstance(x, np.ndarray)):
        return None
    if not (xi.ndim == 1 and yi.ndim == 1 and wi.ndim == 1 and x.ndim == 1):
        return None
    if not (len(xi) == len(yi) == len(wi)):
        return None
    m = len(xi)
    n = len(x)
    result = np.zeros(n)
    for k in range(n):
        numerator = 0.0
        denominator = 0.0
        exact_match = False
        for j in range(m):
            if x[k] == xi[j]:
                result[k] = yi[j]
                exact_match = True
                break
            temp = wi[j] / (x[k] - xi[j])
            numerator += temp * yi[j]
            denominator += temp
        if not exact_match:
            result[k] = numerator / denominator
    return result


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