use std::os::raw::c_void;

type Function = fn(data: *const c_void, n: i64, x: *const f64) -> f64;

#[repr(C)]
struct Closure {
    data: *const c_void,
    function: Function,
}

impl Closure {
    pub fn new<F>(function: &F) -> Closure where F: Fn(&[f64]) -> f64 {
        fn wrap<F>(closure: *const c_void, n: i64, x: *const f64) -> f64
                where F: Fn(&[f64]) -> f64 {
            use std::slice::from_raw_parts;
            let closure = closure as *const F;
            unsafe { (*closure)(from_raw_parts(x, n as usize)) }
        }
        Closure {data: &*function as *const _ as *const c_void, function: wrap::<F>}
    }
}

type FunctionMut = fn(data: *mut c_void, n: i64, x: *const f64) -> f64;

#[repr(C)]
struct ClosureMut {
    data: *mut c_void,
    function: FunctionMut,
}

impl ClosureMut {
    pub fn new<F>(function: &mut F) -> ClosureMut where F: FnMut(&[f64]) -> f64 {
        fn wrap<F>(closure: *mut c_void, n: i64, x: *const f64) -> f64
                where F: FnMut(&[f64]) -> f64 {
            use std::slice::from_raw_parts;
            let closure = closure as *mut F;
            unsafe { (*closure)(from_raw_parts(x, n as usize)) }
        }
        ClosureMut {data: &mut *function as *mut _ as *mut c_void, function: wrap::<F>}
    }
}

extern "C" {
    fn newuoa_closure(function: *mut ClosureMut, n: i64, npt: i64, x: *mut f64,
        rhobeg: f64, rhoend: f64, maxfun: i64, w: *mut f64) -> f64;

    fn newuoa_closure_const(function: *const Closure, n: i64, npt: i64, x: *mut f64,
        rhobeg: f64, rhoend: f64, maxfun: i64, w: *mut f64) -> f64;
}

pub struct Newuoa {
    variables_count: usize,
    number_of_interpolation_conditions: usize,
    initial_trust_region_radius: f64,
    final_trust_region_radius: f64,
    max_function_calls_count: usize,
    working_space: Vec<f64>,
}

impl Newuoa {
    pub fn new() -> Newuoa {
        use std::iter::repeat;
        const VARIABLES_COUNT: usize = 2;
        const NUMBER_OF_INTERPOLATION_CONDITIONS: usize = VARIABLES_COUNT + 2;
        let working_space_size = Newuoa::working_space_size(
            VARIABLES_COUNT,
            NUMBER_OF_INTERPOLATION_CONDITIONS);
        Newuoa {
            variables_count: VARIABLES_COUNT,
            number_of_interpolation_conditions: NUMBER_OF_INTERPOLATION_CONDITIONS,
            initial_trust_region_radius: 1e-6,
            final_trust_region_radius: 1e6,
            max_function_calls_count: 1000,
            working_space: repeat(0.0).take(working_space_size).collect::<_>(),
        }
    }

    pub fn variables_count(&mut self, value: usize) -> &mut Self {
        assert!(value >= 2);
        self.variables_count = value;
        self
    }

    pub fn number_of_interpolation_conditions(&mut self, value: usize) -> &mut Self {
        assert!(value >= 4);
        self.number_of_interpolation_conditions = value;
        self
    }

    pub fn initial_trust_region_radius(&mut self, value: f64) -> &mut Self {
        assert!(value <= self.final_trust_region_radius);
        self.initial_trust_region_radius = value;
        self
    }

    pub fn final_trust_region_radius(&mut self, value: f64) -> &mut Self {
        assert!(value >= self.initial_trust_region_radius);
        self.final_trust_region_radius = value;
        self
    }

    pub fn max_function_calls_count(&mut self, value: usize) -> &mut Self {
        self.max_function_calls_count = value;
        self
    }

    pub fn perform<F>(&mut self, values: &mut [f64], function: &F) -> f64
            where F: Fn(&[f64]) -> f64 {
        self.check(values);
        self.resize_working_space();
        let closure = Closure::new(function);
        unsafe {
            newuoa_closure_const(
                &closure as *const _,
                self.variables_count as i64,
                self.number_of_interpolation_conditions as i64,
                values.as_mut_ptr(),
                self.initial_trust_region_radius,
                self.final_trust_region_radius,
                self.max_function_calls_count as i64,
                self.working_space.as_mut_ptr(),
            )
        }
    }

    pub fn perform_mut<F>(&mut self, values: &mut [f64], function: &mut F) -> f64
            where F: FnMut(&[f64]) -> f64 {
        self.check(values);
        self.resize_working_space();
        let mut closure = ClosureMut::new(function);
        unsafe {
            newuoa_closure(
                &mut closure as *mut _,
                self.variables_count as i64,
                self.number_of_interpolation_conditions as i64,
                values.as_mut_ptr(),
                self.initial_trust_region_radius,
                self.final_trust_region_radius,
                self.max_function_calls_count as i64,
                self.working_space.as_mut_ptr(),
            )
        }
    }

    fn check(&self, values: &[f64]) {
        assert!(values.len() >= self.variables_count);
        assert!(self.number_of_interpolation_conditions >= self.variables_count + 2);
        assert!(self.number_of_interpolation_conditions <=
            (self.variables_count + 1)*(self.variables_count + 2)/2);
    }

    fn resize_working_space(&mut self) {
        use std::iter::repeat;
        let working_space_size = Newuoa::working_space_size(
            self.number_of_interpolation_conditions,
            self.variables_count);
        if self.working_space.len() != working_space_size {
            self.working_space = repeat(0.0).take(working_space_size).collect::<_>();
        }
    }

    fn working_space_size(number_of_interpolation_conditions: usize, variables_count: usize) -> usize {
        3*variables_count*(variables_count + 3)/2
        + (number_of_interpolation_conditions + 13)
            *(number_of_interpolation_conditions + variables_count)
    }
}

#[test]
fn test_pefrorm_mut_with_all_settings_should_succeed() {
    let mut calls_count = Box::new(0);
    let mut values = [10.0, 10.0, 10.0];
    let result = {
        let mut function = |x: &[f64]| -> f64 {
            assert_eq!(x.len(), 3);
            *calls_count += 1;
            x[0]*x[0] + x[1]*x[1] + x[2]*x[2]
        };
        Newuoa::new()
            .variables_count(values.len())
            .number_of_interpolation_conditions((values.len() + 1)*(values.len() + 2)/2)
            .initial_trust_region_radius(1e-3)
            .final_trust_region_radius(1e3)
            .max_function_calls_count(25)
            .perform_mut(&mut values, &mut function)
    };
    for x in values.iter() {
        assert!(*x <= 1e-3);
    }
    assert!(result <= 1e-3);
    assert_eq!(*calls_count, 25);
}
