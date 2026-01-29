import numpy as np
import sympy
from uncertainties import ufloat
from uncertainties.umath import *

def mass_function(df):
    A,B,C,F,G,H = df[['a_thiele_innes','b_thiele_innes','c_thiele_innes','f_thiele_innes','g_thiele_innes','h_thiele_innes']]
    A,B,C,F,G,H = float(A), float(B), float(C), float(F), float(G), float(H)
    u = 0.5 * (A**2 + B**2 + F**2 + G**2)
    v = A * G - B * F
    a0_mas = np.sqrt(u + np.sqrt(u**2 - v**2))
    fm = float((a0_mas**3 * 365.25**2) / (df["period"]**2 * df["parallax"]**3))
    return fm

def a_conversion(m1, m2, f):
    q = m2/m1
    return q/(1+q) - f/(1+f)

def get_phot_a(period, m1, m2, f, phot=True):
    '''
    calculate the photocenter semi-major axis of a binary adapted from gaiamock
    period: orbital period in days
    m1: mass of star 1 in Msun
    m2: mass of star 2 in Msun
    f = F2/F1 is flux ratio in G band. 
    '''
    G, Msun, AU = 6.6743e-11, 1.98840987069805e+30, 1.4959787e+11
    a_au = (((period*86400)**2 * G * (m1*Msun + m2*Msun)/(4*np.pi**2))**(1/3.))/AU
    if not phot: # spit out the physical semimajor axis
        return a_au
    a_au = a_au*a_conversion(m1, m2, f)
    return a_au

def get_p_from_a(a_au, m1, m2, f, phot=False):
    '''
        get period in days
    '''
    G, Msun, AU = 6.6743e-11, 1.98840987069805e+30, 1.4959787e+11
    a_com = a_au
    if phot: # rescale semi-major axis if input is photocentre
        a_com = a_au/a_conversion(m1, m2, f)
    period = ((a_com*AU)**3 * (4*np.pi**2)/(m1*Msun + m2*Msun) / G)**(1/2)/86400
    return period

def mass_function_explicit(period, m1, m2, f):
    '''
        period in days
        m1, m2 in msolar
        f = f2/f1
    '''
    a_au = get_phot_a(period, m1, m2, f)
    return a_au**3 * (period/365.25)**(-2)

def mass_function_explicit_a(a_au, m1, m2, f):
    period = get_p_from_a(a_au, m1, m2, f)
    return (a_au*a_conversion(m1,m2,f))**3 * (period/365.25)**(-2)

def mass_function_reduced(m1,m2,f):
    # turns out only the masses matter
    G, Msun, AU = 6.6743e-11, 1.98840987069805e+30, 1.4959787e+11
    return (86400*365.25)**2 * G / (4*np.pi**2) / AU**3 * Msun * (m1 + m2) * a_conversion(m1, m2, f)**3

def companion_mass(m1,fm):
    G, Msun, AU = 6.6743e-11, 1.98840987069805e+30, 1.4959787e+11
    C = (86400*365.25)**2 * G / (4*np.pi**2) / AU**3 * Msun
    
    m = sympy.symbols("m", real=True) # guarantee real solutions
    m1 = float(m1)
    roots = sympy.solve((C/fm)*m**3 - m**2 - m1**2 - 2*m*m1)
    try:
        return float(np.max(roots))
    except:
        return 0

def mass_function_error(df):
    A = ufloat(df['a_thiele_innes'], df['a_thiele_innes_error'])
    B = ufloat(df['b_thiele_innes'], df['b_thiele_innes_error'])
    F = ufloat(df['f_thiele_innes'], df['f_thiele_innes_error'])
    G = ufloat(df['g_thiele_innes'], df['g_thiele_innes_error'])
    u = 0.5 * (A**2 + B**2 + F**2 + G**2)
    v = A * G - B * F
    a0_mas = (u + (u**2 - v**2)**(1/2))**(1/2)
    
    period = ufloat(df['period'], df['period_error'])
    parallax = ufloat(df['parallax'], df['parallax_error'])
    fm = (a0_mas**3 * 365.25**2) / (period**2 * parallax**3)
    return fm

def mass_function_error_quick(df):
    a0_mas = ufloat(df['a0_mas'], df['a0_mas_err'])
    period = ufloat(df['period'], df['period_error'])
    parallax = ufloat(df['parallax'], df['parallax_error'])
    fm = (a0_mas**3 * 365.25**2) / (period**2 * parallax**3)
    return fm

def thiele_innes(df):
    return df[['a_thiele_innes','b_thiele_innes','f_thiele_innes','g_thiele_innes']]

def calculate_inc(df):
    A,B,F,G = thiele_innes(df)
    return (A*G - B*F) / ( (A**2 + B**2) * (F**2 + G**2) )