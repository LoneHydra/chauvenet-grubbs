import json
import argparse
import random
import math
from scipy.stats import norm
from scipy.stats import t
import pyreadstat
import pandas as pd

# Ładowanie danych z pliku JSON
def load_from_json(filename):
    with open(filename, 'r') as f:
        data = json.load(f)
    
    # Walidacja danych - czy jest to array liczb
    if isinstance(data, list) and all(isinstance(item, (int, float)) for item in data):
        return data
    else:
        raise ValueError("The JSON file must contain an array of numbers.")

# Ładowanie danych z pliku CSV
def load_from_csv(filename):
    data = pd.read_csv(filename, header=None)
    if data.shape[1] == 1:
        return data.iloc[:, 0].tolist()
    else:
        raise ValueError("The CSV file must contain a single column of numbers.")

# Work in progress - ładowanie z jednokolumnowego pliku SAV
def load_from_sav(filename):
    data = pd.read_spss(filename)
    if data.shape[1] == 1:
        return data.iloc[:, 0].tolist()
    else:
        raise ValueError("The SAV file must contain a single column of numbers.")

def load_data(filename):
    if filename.endswith('.json'):
        return load_from_json(filename)
    elif filename.endswith('.csv'):
        return load_from_csv(filename)
    elif filename.endswith('.sav'):
        return load_from_sav(filename)
    else:
        raise ValueError("Unsupported file format. Please provide a .json, .csv, or .sav file.")

# Stworzenie losowych danych
def generate_random_sample(size, min_val, max_val):
    return [random.randint(min_val, max_val) for _ in range(size)]

# Sprawdzenie kryterium Chauveneta i test Grubbsa
def chauvenet(data_array, x, alpha):
    a = 0
    N = len(data_array) # Ilość liczb w zbiorze
    print("Sample size: ", N)

    # Obliczenie średniej arytmetycznej
    for number in data_array:
        a += number
    avg = a/N
    print ("Average: ", avg) 

    # Obliczenie estymatora największej wiarygodności
    b = 0
    for number in data_array:
        b += (number - avg) ** 2 # Zmienna pomocniczna (suma dla każdego [x - średnia]^2)

    sigma = math.sqrt(1/(N - 1) * b) 
    print ("Sigma: ", sigma)

    k = abs(avg - x)/sigma # Obliczenie współczynnika k
    print ("Indicator k: ", k)

    prob = 2 * (1 - norm.cdf(k)) # Prawdopodobieństwo dla współczynnika k
    print("Probability of k:", prob)

    if prob < (1 / (2 * N)): # Sprawdzenie czy prawdopodobieństwo jest mniejsze niż 1/2N
        print(f"The value {x} is an outlier according to the Chauvenet criterion.")
    else:
        print(f"The value {x} is not an outlier according to the Chauvenet criterion.")

    # Test Grubbsa
    G = max(abs(number - avg) for number in data_array) / sigma # Obliczenie wartości G
    print("G: ", G)
    
    t_crit = t.ppf(q = (alpha / (2 * N)), df = (N - 2)) # Obliczenie wartości krytycznej dystrybucji t
    G_critical = ((N - 1) / math.sqrt(N)) * (math.sqrt(t_crit**2 / (N - 2 + t_crit**2))) # Obliczenie wartości krytycznej G zgodnie ze wzorem

    print(f"Critical G value: {G_critical:.4f}")
    print("Alpha: ", alpha)

    if G > G_critical: 
        print(f"The value {x} is an outlier according to the two-sided Grubbs' test.")
    else:
        print(f"The value {x} is not an outlier according to the two-sided Grubbs' test.")
    

# Ustalenie argumentów
parser = argparse.ArgumentParser(description="Process a JSON file or generate random data.")
parser.add_argument('-x', type=float, required=True, help="Value of x to be used in the script.")
parser.add_argument('-a', '--alpha', type=float, required=False, help="Value of alpha to be used in the script.")
parser.add_argument('-p', '--path', type=str, help="Path to the JSON file.")
parser.add_argument('-r', '--random', type=int, nargs=3, metavar=('SIZE', 'MIN', 'MAX'),
                    help="Generate random data: SIZE is the number of random numbers, "
                         "MIN is the minimum value, MAX is the maximum value.")

args = parser.parse_args()

# Oddelegowanie do funkcji
if args.alpha:
    alpha = args.alpha
else:
    alpha = 0.05 # Domyślna wartość alpha 0.05 jeśli nie podano innej
if args.path:
    try:
        data_array = load_data(args.path)
        x = args.x
        chauvenet(data_array, args.x, alpha)
    except FileNotFoundError:
        print("File not found. Ensure the path is correct.")
    except json.JSONDecodeError:
        print("Failed to decode JSON. Ensure the file is properly formatted.")
elif args.random:
    size, min_val, max_val = args.random
    data_array = generate_random_sample(size, min_val, max_val)
    chauvenet(data_array, args.x, alpha)
else:
    print("You must provide the value of -x and either -p for a JSON file path or -r to generate random data.")

