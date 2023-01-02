import seaborn
# colors to use for each class throughout the report
pal = seaborn.color_palette("pastel", 5)
class_0 = t_shirt_color = pal[0]
class_1 = trousers_color = pal[1]
class_2 = pullover_color = pal[2]
class_3 = dress_color = pal[3]
class_4 = shirt_color = pal[4]


def map_cls_to_clothing():
    return {0: "T-shirt/Top", 1: "Trousers", 2: "Pullover",
            3: "Dress", 4: "Shirt"}
