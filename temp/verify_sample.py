import math

def H(pos, neg):
    t = pos+neg
    if t==0: return 0
    p = pos/t
    n = neg/t
    return -(p*math.log2(p) if p>0 else 0) - (n*math.log2(n) if n>0 else 0)

# Total 14: 9 Yes, 5 No
print("Total H:", H(9, 5))

# Outlook
# Sunny: 2Y, 3N
# Overcast: 4Y, 0N
# Rain: 3Y, 2N
print("Sunny H:", H(2,3))
print("Overcast H:", H(4,0))
print("Rain H:", H(3,2))
IG_out = H(9,5) - (5/14*H(2,3) + 4/14*H(4,0) + 5/14*H(3,2))
print("IG Outlook:", IG_out)

# Temperature
# Hot: 2Y, 2N
# Mild: 4Y, 2N
# Cool: 3Y, 1N
print("Hot H:", H(2,2))
print("Mild H:", H(4,2))
print("Cool H:", H(3,1))
IG_temp = H(9,5) - (4/14*H(2,2) + 6/14*H(4,2) + 4/14*H(3,1))
print("IG Temp:", IG_temp)

# Humidity
# High: 3Y, 4N
# Normal: 6Y, 1N
print("High H:", H(3,4))
print("Normal H:", H(6,1))
IG_hum = H(9,5) - (7/14*H(3,4) + 7/14*H(6,1))
print("IG Humidity:", IG_hum)

# Wind
# Weak: 6Y, 2N
# Strong: 3Y, 3N
print("Weak H:", H(6,2))
print("Strong H:", H(3,3))
IG_wind = H(9,5) - (8/14*H(6,2) + 6/14*H(3,3))
print("IG Wind:", IG_wind)
