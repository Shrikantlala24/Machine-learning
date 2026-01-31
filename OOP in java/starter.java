class starter {
    public static void main(String args[]) {
        
        // int x = 10;
        Car C1 = new Car("Toyota", 2020);
        System.out.println(C1.name);


    }
}

class Car {
    String name;
    // String Brand;
    // String color;
    int year;

    Car(String n, int y) {
        this.name = n;
        this.year = y;
    }
}