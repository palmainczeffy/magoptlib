import json


def text_to_array(input_file):
    """ Convert the input file to arrays.

    Fills up b_values, phi_values, and theta_values from a json file.

    Parameters
    ----------

    input_file : str
        The input file in json format.

    Returns
    -------
    b_values : list
        The spherical harmonics signal, in our case the magnetic field values.

    phi_values : list
        The azimuthal angle values.

    theta_values : list
        The polar angle values.
    """

    if not isinstance(input_file, str):
        raise TypeError("Input data should be a JSON string.")

    try:
        with open(input_file, "r") as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        raise TypeError("Input data is not valid JSON.") from e

    if not isinstance(data, dict):
        raise TypeError("The JSON data should be a dictionary.")

    b_values = []
    phi_values = []
    theta_values = []

    n = len(data['data'])
    for i in range(n):

        entry = data['data'][i]

        if 'value' not in entry or 'phi' not in entry or 'theta' not in entry:
            raise KeyError("Dictionary must contain 'value', 'phi', and 'theta' keys.")

        b_values.append(entry['value'])
        phi_values.append(entry['phi'])
        theta_values.append(entry['theta'])

    return b_values, phi_values, theta_values
