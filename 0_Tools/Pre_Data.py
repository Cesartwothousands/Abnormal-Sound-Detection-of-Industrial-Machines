Data_Fan = []
Data_Pump = []
Data_Slider = []
Data_Valve = []

# The number of files
num_v0a = 119
num_v0n = num_v0a + 991
num_v2a = num_v0n + 120
num_v2n = num_v2a + 708
num_v4a = num_v2n + 120
num_v4n = num_v4a + 1000
num_v6a = num_v4n + 120
num_v6n = num_v6a + 992
print('number of Valve = ' + str(num_v6n))
num_s0a = num_v6n + 356
num_s0n = num_s0a + 1068
num_s2a = num_s0n + 267
num_s2n = num_s2a + 1068
num_s4a = num_s2n + 178
num_s4n = num_s4a + 534
num_s6a = num_s4n + 89
num_s6n = num_s6a + 534
print('number of Slider = ' + str(num_s6n - num_v6n ))
num_p0a = num_s6n + 143
num_p0n = num_p0a + 1006
num_p2a = num_p0n + 111
num_p2n = num_p2a + 1005
num_p4a = num_p2n + 100
num_p4n = num_p4a + 702
num_p6a = num_p4n + 102
num_p6n = num_p6a + 1036
print('number of Pump = ' + str(num_p6n - num_s6n ))
num_f0a = num_p6n + 407
num_f0n = num_f0a + 1011
num_f2a = num_f0n + 359
num_f2n = num_f2a + 1016
num_f4a = num_f2n + 348
num_f4n = num_f4a + 1033
num_f6a = num_f4n + 361
num_f6n = num_f6a + 1015
print('number of Fan = ' + str(num_f6n - num_p6n ))
num_total = num_f6n
print('number of total = ' + str(num_total))