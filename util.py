from math import radians, sin, cos, atan2, sqrt

def distancia_haversine(lat1, lng1, lat2, lng2):
    R = 6371
    lat1, lng1, lat2, lng2 = map(radians, [lat1, lng1, lat2, lng2])
    dlat = lat2 - lat1
    dlng = lng2 - lng1
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlng/2)**2
    return R * 2 * atan2(sqrt(a), sqrt(1 - a))

def distancia_euclidiana(x1,y1,x2,y2):
    dx = x2 - x1
    dy = y2 - y1
    return sqrt(pow(dx, 2) + pow(dy,2))