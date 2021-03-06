extern crate newuoa;

fn main() {
    let mut calls_count = Box::new(0);
    let mut values = vec![0.0_f64, - 5.0_f64.sqrt()];
    println!("initial: {:?}", values);
    let result = {
        let mut function = |x: &[f64]| -> f64 {
            assert!(x.len() == 2);
            *calls_count += 1;
            -4.0*x[0]*x[1] + 5.0*x[0]*x[0] + 8.0*x[1]*x[1]
                + 16.0*(5.0_f64).sqrt()*x[0] + 8.0*(5.0_f64).sqrt()*x[1] - 44.0
        };
        newuoa::Newuoa::new()
            .variables_count(2)
            .number_of_interpolation_conditions((2 + 1)*(2 + 2)/2)
            .initial_trust_region_radius(1e-3)
            .final_trust_region_radius(1e3)
            .max_function_calls_count(100)
            .perform_mut(&mut values, &mut function)
    };
    println!("final: {:?}", values);
    println!("result: {}", result);
    println!("calls_count: {}", *calls_count);
}
