import camara as Shop
def main():
    class_shop=Shop.ShopIA()
    
    cap=class_shop.init()
    
    stream=class_shop.tiendaIA(cap)
    
if __name__=="__main__":
    main()