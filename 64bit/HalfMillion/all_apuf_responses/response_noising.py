import numpy as np
inter_bit = 8
clean_response = np.loadtxt('./1resp_XOR_APUF_chal_64_500000.csv', delimiter=',')
print(clean_response)
for i in range(0, 9999, inter_bit):
    if clean_response[i] == 0:
        clean_response[i] = 1
    else:
        clean_response[i] = 0

print(type(clean_response))

b = np.savetxt('./response_noised.csv', clean_response, fmt='%d', delimiter=',')
