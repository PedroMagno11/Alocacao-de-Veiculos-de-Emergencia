import folium

pontos = [
    (-23.277331071506687, -45.94892451480434),
    (-23.341287868696867,-46.032481113318205),
    (-23.31446167900042 , -45.9801926857208),
    (-23.304717003281688 ,  -45.92333485380797),
    (-23.2920464748932 , -46.032891316847284)
]

lat_media = sum(p[0] for p in pontos) / len(pontos)
lon_media = sum(p[1] for p in pontos) / len(pontos)

mapa = folium.Map(location=[lat_media, lon_media], zoom_start=14)

for i, (lat, lon) in enumerate(pontos):
    folium.Marker(
        location=[lat, lon],
        popup=f"Ponto {i+1}",
        tooltip=f"Ponto {i+1}"
    ).add_to(mapa)

mapa.save("carta_pontos.html")