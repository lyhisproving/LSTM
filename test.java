int x,y;
int a = input();
int b = input();

x = a+b;
if(a>0){
    y=1;
}else{
    y=-1;
}
public class EmployeeTest extends TestCase {
    String firstName, lastName, ssn;
    double baseSalary, commissionRate = 0.5, grossSales;
    Employee E = new Employee(firstName, lastName, ssn,
    baseSalary, commissionRate,grossSales);
    ···
}

public class EmployeeTest extends TestCase {
    String firstName, lastName, ssn;
    double baseSalary, commissionRate = 0.5, grossSales;
    Employee E = new Employee(firstName, lastName, ssn,
    baseSalary, commissionRate,
    grossSales);
    
    public void testToString() {
        E.setFirstName("John")
        E.setGrossSales(200);
        E.setBaseSalary(100);
        String expected = "Employee: null\n" +
        "social security number: null\n" +
        "total salary: 200.00";
        assertEquals(E.toString(), expected);
    }
    ···
}
