import json


def load_text(input_file):
    """ Load text from the input file.

    Parameters
    ----------
    input_file : str
        The input file in json format.

    Returns
    -------
    data : dict
        The data from the input file, in a dictionary format.

    """
    with open(input_file, "r") as f: 
        data = json.load(f)
    return data

def text_to_array(data):
    """ Convert the dictionary to arrays.

    Fills up the b_values, phi_values, and theta_values arrays from the dictionary.

    Parameters
    ----------

    data : dict
        The data from the input file, in a dictionary format.

    Returns
    -------
    b_values : list
        The spherical harmonics signal, in our case the magnetic field values.

    phi_values : list
        The azimuthal angle values.  

    theta_values : list
        The polar angle values.  

    """
    b_values =[]
    phi_values = []
    theta_values = []

    n=len(data['data'])
    for i in range(n):
        b_values.append(data['data'][i]['value'])
        phi_values.append(data['data'][i]['phi'])
        theta_values.append(data['data'][i]['theta'])

    return b_values, phi_values, theta_values