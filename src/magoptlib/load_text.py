import json


def load_text(input_file):
    with open(input_file, "r") as f: 
        data = json.load(f)
    return data

def text_to_array(data):
    b_values =[]
    phi_values = []
    theta_values = []

    n=len(data['data'])
    for i in range(n):
        b_values.append(data['data'][i]['value'])
        phi_values.append(data['data'][i]['phi'])
        theta_values.append(data['data'][i]['theta'])

    return b_values, phi_values, theta_values