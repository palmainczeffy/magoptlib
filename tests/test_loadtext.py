from magoptlib import magoptlib 
from magoptlib import load_text
from magoptlib.load_text import load_text, text_to_array
from magoptlib.magoptlib import get_sh_coeff, modified_sh
import numpy as np


data=load_text("tests/test_simulation_cubic_magnet_12mm_3_randompolarisation.mag.pkl.mag.json")
b_values, phi_values, theta_values = text_to_array(data)

print(b_values)
print(phi_values)
print(theta_values)

print(np.asarray(b_values).shape[0])
print(np.asarray(phi_values).shape[0])
print(np.asarray(theta_values).shape[0])

assert np.allclose(np.asarray(b_values).shape[0], 648),  "Test failed"