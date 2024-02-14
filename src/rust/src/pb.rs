#![allow(dead_code)]

//! `simple_bar` is an extremely simple terminal progress bar
//! 
//! # Example
//! 
//! ```
//! use std::{thread::sleep, time::Duration};
//! use simple_bar::ProgressBar;
//! 
//! let num_iterations = 500;
//! let mut bar = ProgressBar::default(num_iterations, 50, false);
//! 
//! for _ in 0..num_iterations {
//!     bar.update();
//!     sleep(Duration::from_millis(1));
//! }
//! ```
//! 
//! This example generates the following output:
//! 
//! ![above code generates](https://mie-res.netlify.app/simple_bar_example.png)

use std::time::{Instant, Duration};
use std::io::{stderr, Write};



#[derive(Debug, Clone, Copy)]
struct Eta {
    last_time: Instant,
    last_step: u32,
    total: u32,
    eta: Duration,
    latest_speeds: [f32; 10],
    curr_idx: usize,
    done_once: bool,
    it_s: f32,
}

impl Eta {
    pub fn new(total: u32) -> Eta {
        Eta { 
            last_time: Instant::now(),
            last_step: 0,
            total,
            eta: Duration::MAX,
            latest_speeds: [f32::MIN; 10],
            curr_idx: 0,
            done_once: false,
            it_s: 0.0,
        }
    }

    fn should_update(&self) -> bool {
        !self.done_once || Instant::now() - self.last_time >= Duration::from_secs(1)
    }

    fn update(&mut self, step: u32) {
        self.update_latest_speeds(step);
        self.update_last_step(step);
        self.update_last_time();
        self.update_it_s();

        let speed = self.get_mean_speed();

        if speed > 0.0 {
            let to_go = (self.total - step) as f32 / speed;
            self.eta = Duration::from_millis(to_go as u64);
        }
    }

    fn get_steps_taken(&self, step: u32) -> u32 {
        step - self.last_step
    }

    fn get_elapsed_time(&self) -> Duration {
        Instant::now() - self.last_time
    }

    fn update_last_time(&mut self) {
        self.last_time = Instant::now();
    }

    fn update_last_step(&mut self, step: u32) {
        self.last_step = step;
    }

    fn update_latest_speeds(&mut self, step: u32) {
        let steps_taken = self.get_steps_taken(step);
        let elapsed_time = self.get_elapsed_time();

        self.latest_speeds[self.curr_idx] = steps_taken as f32 / elapsed_time.as_millis() as f32;
        self.curr_idx += 1;
        self.curr_idx %= 10;

        if !self.done_once && self.curr_idx == 0 {
            self.done_once = true;
        }
    }

    fn update_it_s(&mut self) {
        self.it_s = self.get_mean_speed();
    }

    fn get_mean_speed(&self) -> f32 {
        let mut speed = 0.0;

        for val in self.latest_speeds {
            speed += val;
        }

        speed / 10.0
    }

    pub fn get_eta(&mut self, step: u32) -> Duration {
        if self.should_update() {
            self.update(step)
        }
        self.eta
    }
}

pub struct ProgressBar {
    delimiters: (char, char),
    num_iterations: u32,
    length: u32,
    state: u32,
    progress_char: char,
    last_progress_char: char,
    empty_char: char,
    eta: Option<Eta>,
}

impl ProgressBar {
    /// Creates a new ProgressBar given:
    /// 1. `num_iterations: u32` the number of iterations
    /// 2. `progress_char: char` the `char` to be printed on the completed spots of the progress bar
    /// 3. `last_progress_char: char` the `char` to be printed as the last of the progess ones (e.g '>' to make '==>')
    /// 4. `empty_char: char` the `char` to be printed on the empty spots of the progress bar
    pub fn new(
        delimiters: (char, char),
        num_iterations: u32,
        length: u32,
        progress_char: char,
        last_progress_char: char,
        empty_char: char,
        eta: bool
    ) -> ProgressBar 
    {
        let mut b_eta = None;
        if eta {
            b_eta = Some(Eta::new(num_iterations));
        }
        ProgressBar { delimiters, num_iterations, length, state: 0, progress_char, last_progress_char, empty_char, eta: b_eta }
    }

    /// Creates a new ProgressBar with the default `char`s for the completed and empty spots of the
    /// progress bar, which are: `'█'` and `' '` respectively.
    pub fn default(num_iterations: u32, length: u32, eta: bool) -> ProgressBar {
        ProgressBar::new(('[', ']'), num_iterations, length, '█', '█', ' ', eta)
    }

    ///Creates a new ProgressBar using the cargo style (e.g. [===>  ])
    pub fn cargo_style(num_iterations: u32, length: u32, eta: bool) -> ProgressBar {
        ProgressBar::new( ('[', ']'), num_iterations, length, '=', '>', ' ', eta)
    }

    /// Changes the `char`s for the completed, last progress, and empty spots of the progress bar.
    pub fn reformat(&mut self, delimiters: (char, char), progress_char: char, last_progress_char: char, empty_char: char) {
        self.delimiters = delimiters;
        self.progress_char = progress_char;
        self.last_progress_char = last_progress_char;
        self.empty_char = empty_char;
    }

    pub fn reset(&mut self) {
        self.state = 0;
        match self.eta {
            Some(_) => { self.eta = Some(Eta::new(self.num_iterations)); }
            None => { }
        };
    }

    fn print_bar(&mut self, perc: u32){
        let length_dig = (self.num_iterations as f64).log10() as i32;
        let state_dig = (self.state as f64).log10() as i32;
        let ratio = remap(self.state as f32, 0., self.num_iterations as f32, 1., self.length as f32);

        let mut terminal_string = String::from("");

        print!("\r");

        for _ in 0..(length_dig - state_dig) {
            terminal_string = format!("{}{}", terminal_string, 0);
        }
        terminal_string = format!("{}{} / {} {}", terminal_string, self.state, self.num_iterations, self.delimiters.0);
        for _ in 0..ratio-1 {
            terminal_string = format!("{}{}", terminal_string, self.progress_char);
        }
        if self.is_last() {
            terminal_string = format!("{}{}", terminal_string, self.progress_char);
        } else {
            terminal_string = format!("{}{}", terminal_string, self.last_progress_char);
        }
        for _ in ratio..self.length {
            terminal_string = format!("{}{}", terminal_string, self.empty_char);
        }
        terminal_string = format!("{}{} ({}%)", terminal_string, self.delimiters.1, perc);

        match self.eta {
            None => (),
            Some(mut e) => {
                let sec_total = e.get_eta(self.state).as_secs();
                self.eta = Some(e);
                let hours = sec_total / 3600;
                let mins = ( sec_total / 60 ) % 60;
                let secs = sec_total % 60;
                if hours > 99 {
                    terminal_string = format!("{} ETA ??:??:??", terminal_string);
                } else {
                    let hours_digits = (hours/10, hours%10);
                    let mins_digits = (mins/10, mins%10);
                    let sec_digits = (secs/10, secs%10);

                    terminal_string = format!("{} ETA {}{}:{}{}:{}{}",
                        terminal_string,
                        hours_digits.0,
                        hours_digits.1,
                        mins_digits.0,
                        mins_digits.1,
                        sec_digits.0,
                        sec_digits.1);
                }
                terminal_string = format!("{} it/s {:.2}", terminal_string, e.it_s);
            }
        }
        
        print!("{}", terminal_string);

        if self.is_last() {
            println!("");
        }

        stderr().flush().unwrap();
    }

    /// Updates the progress bar
    pub fn update(&mut self) {
        // assert!(self.state < self.num_iterations + 1 );
        self.state += 1;
        let perc = ((self.state as f32 / self.num_iterations as f32) * 100.) as u32;
        self.print_bar(perc);
    }

    fn is_last(&self) -> bool {
        self.state == self.num_iterations
    }
}

fn remap(val: f32, min: f32, max: f32, new_min: f32, new_max: f32) -> u32 {
    (new_min + (val - min) * (new_max - new_min) / (max - min)) as u32
}