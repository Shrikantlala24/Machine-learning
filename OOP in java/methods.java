public class methods {
    public static void main(String[] args) {
        car myCar = new car();
        myCar.model = "Toyota";
        myCar.color = "Red";
        myCar.year = 2020;

        myCar.honk();

    }
}
class car {
    String model;
    String color;
    int year;

    public void start() {
        System.out.println("Car is starting");
    }

    public void stop() {
        System.out.println("Car is stopping");
    }

    static void honk() {
        System.out.println("Car is honking");
    }
}