use rodio::source::{SineWave,Delay};
use rodio::{OutputStream, Sink, Source};
use anyhow::{Context, Result};
use anyhow::anyhow;
use std::time::Duration;
use std::thread;
use std::sync::{Arc, Mutex};
use rand::Rng;
use sdl2::keyboard::Keycode;
use sdl2::event::Event;
use std::fs::File;
use std::io::Read;
use std::path::Path;
use sdl2::pixels::Color;
use sdl2::rect::Rect;
use sdl2::render::Canvas;
use sdl2::video::Window;
use std::process;
use std::env;
use rodio::source::Amplify;
use rodio::source::TakeDuration;

fn cpu_init() -> ([u8; 0x1000], [u8; 16], u16, u8, u8, u16, u8, [u16; 16], [[u8; 64]; 32]) {
    let memory: [u8; 0x1000] = [0; 0x1000]; // Memory
    let vx: [u8; 16] = [0; 16]; // General registry. VF is saved for flags
    let i: u16 = 0; // Address registry
    let delay: u8 = 0; // Delay timer
    let sound: u8 = 0; // Sound timer
    let pc: u16 = 0x200; // Program counter
    let sp: u8 = 0; // Stack pointer
    let stack: [u16; 16] = [0; 16]; // 16 x 16bit stack allowing for 16 nested subroutines
    let display: [[u8; 64]; 32] = [[0; 64]; 32]; //  [[u8; COLUMN]; ROW] accessed with display[ROW][COL]
    (memory, vx, i, delay, sound, pc, sp, stack, display)
}

fn font_init() -> [u8; 80] {
    [ 
    0xF0, 0x90, 0x90, 0x90, 0xF0, // 0
    0x20, 0x60, 0x20, 0x20, 0x70, // 1
    0xF0, 0x10, 0xF0, 0x80, 0xF0, // 2
    0xF0, 0x10, 0xF0, 0x10, 0xF0, // 3
    0x90, 0x90, 0xF0, 0x10, 0x10, // 4
    0xF0, 0x80, 0xF0, 0x10, 0xF0, // 5
    0xF0, 0x80, 0xF0, 0x90, 0xF0, // 6
    0xF0, 0x10, 0x20, 0x40, 0x40, // 7
    0xF0, 0x90, 0xF0, 0x90, 0xF0, // 8
    0xF0, 0x90, 0xF0, 0x10, 0xF0, // 9
    0xF0, 0x90, 0xF0, 0xF9, 0x90, // A
    0xE0, 0x90, 0xE0, 0x90, 0xE0, // B
    0xF0, 0x80, 0x80, 0x80, 0xF0, // C
    0xE0, 0x90, 0x90, 0x90, 0xE0, // D
    0xF0, 0x80, 0xF0, 0x80, 0xF0, // E
    0xF0, 0x80, 0xF0, 0x80, 0x80  // F
    ]
}

// 
fn rom_init(file_path: &str) -> Result<Vec<u8>> {
    let mut file = File::open(file_path).with_context(|| format!("Failed to open the ROM file at {}", file_path))?;
    let mut rom_data = Vec::new();
    file.read_to_end(&mut rom_data).with_context(|| format!("Failed to read ROM file {}", file_path))?;
    Ok(rom_data)
}

fn sound_init() -> (Arc<Mutex<Sink>>, Amplify<TakeDuration<SineWave>>, Delay<Amplify<TakeDuration<SineWave>>>) {
    let (_stream, stream_handle) = OutputStream::try_default().unwrap();
    let sink = Sink::try_new(&stream_handle).unwrap();
    let source = SineWave::new(440 as u32)
        .take_duration(Duration::from_millis(300))
        .amplify(0.3);
    let delay = source.clone().delay(Duration::from_millis(600));
    (Arc::new(Mutex::new(sink)), source, delay) // Locking my sink in a mutex so that I can use it within the thread 
}

// Cloning sink so I don't transfer ownership 
fn play_sound(sink: Arc<Mutex<Sink>>, source: Amplify<TakeDuration<SineWave>>, delay: Delay<Amplify<TakeDuration<SineWave>>>) {
    let sink_clone = sink.clone();
    thread::spawn(move || {
        let locked_sink = sink_clone.lock().unwrap();
        locked_sink.append(source);
        locked_sink.append(delay);
    });
}

fn display_init(scale: u32) -> Result<(Canvas<Window>, sdl2::Sdl)> {
    let sdl_context = sdl2::init().map_err(|e| anyhow::anyhow!(e))?;
    let video_subsystem = sdl_context.video().map_err(|e| anyhow::anyhow!(e))?;
    let window = video_subsystem
        .window("Chip-8 Emulator", 64 * scale, 32 * scale)
        .position_centered()
        .opengl()
        .build()?;
    let canvas = window.into_canvas().build()?;
    Ok((canvas, sdl_context))
}

fn render_display(canvas: &mut Canvas<Window>, display: &[[u8; 64]; 32], scale:u32) -> Result<()> {
    canvas.set_draw_color(Color::BLACK);
    canvas.clear();

    canvas.set_draw_color(Color::WHITE);
    for (row, row_pixels) in display.iter().enumerate() { 
        for (col, &pixel) in row_pixels.iter().enumerate() {
            if pixel == 1{
                let x = (col as u32) * scale;
                let y = (row as u32) * scale;
                let rect = Rect::new(x as i32, y as i32, scale, scale);
                canvas.fill_rect(rect).map_err(|e| anyhow::anyhow!(e))?;
            }
        }
    }
    canvas.present();
    Ok(())
}

fn main() -> Result<(), anyhow::Error> {
    // Possible files are: [,
    //                      15puzzle, pong, pong(1p), tetris, ...
    //                      ]

    let args: Vec<String> = env::args().skip(1).collect();
    if args.is_empty() {
        println!("Call with filename as an argument without extension. \n ROMs are stored in /roms/");
        process::exit(1);
    }
    let (mut memory, mut vx, mut i, mut delay, mut sound, mut pc, mut sp, mut stack, mut display)  = cpu_init();  // Initializing the chip-8
    
    let filen: String = args[0].to_owned().to_ascii_uppercase();
    let dir = Path::new("roms");
    let file_path = dir.join(format!("{}.ch8", filen)); 
    if file_path.exists() {
        let rom_data = rom_init(file_path.to_str().unwrap()).expect("Failed to load ROM file");
        memory[0x200..0x200 + rom_data.len()].copy_from_slice(&rom_data);
    } else {
        eprintln!("ROM file not found: {}", file_path.display());
        process::exit(1);
    }

    memory[0..80].copy_from_slice(&font_init()); // Storing Font data in 0x000:0x1FF 
    let (sink, source, sin_delay) = sound_init(); // Initializing the sound
    // *
    // * PC is init to 0x200, and grabs the opcode stored at 0x200/0x201. Decodes this as big-endian. Thus, can enter into the running loop by init to 0x200
    // *
    let refresh: u16= 60; // 60Hz refresh rate
    let mut rng = rand::thread_rng(); // For rand num gen

    let scale = 10; 
    let (mut canvas, sdl_context) = display_init(scale)?; // initializing the display
    let mut running = true; // starting the game

    let mut event_pump = sdl_context.event_pump().unwrap();

    let mut keypad = [false; 16];  // Keypad. Mapping in Outline.txt
    let mut last_timer_update = std::time::Instant::now(); // For time update

    while running {
        // Handling keyboard input at the start of the loop
        for event in event_pump.poll_iter() {
            match event {
                Event::Quit { .. } => {break},
                Event::KeyDown { keycode: Some(keycode), .. }  => {
                    match keycode {
                        Keycode::Num1 => keypad[0x1] = true,
                        Keycode::Num2 => keypad[0x2] = true,
                        Keycode::Num3 => keypad[0x3] = true,
                        Keycode::Num4 => keypad[0xC] = true,
                        Keycode::Q => keypad[0x4] = true,
                        Keycode::W => keypad[0x5] = true,
                        Keycode::E => keypad[0x6] = true,
                        Keycode::R => keypad[0xD] = true,
                        Keycode::A => keypad[0x7] = true,
                        Keycode::S => keypad[0x8] = true,
                        Keycode::D => keypad[0x9] = true,
                        Keycode::F => keypad[0xE] = true,
                        Keycode::Z => keypad[0xA] = true,
                        Keycode::X => keypad[0x0] = true,
                        Keycode::C => keypad[0xB] = true,
                        Keycode::V => keypad[0xF] = true,
                        Keycode::Escape => {
                            running = false;
                            break;
                        }
                        _ => {()},
                    }
                },
                Event::KeyUp { keycode: Some(keycode), .. } => {
                    match keycode {
                        Keycode::Num1 => keypad[0x1] = false,
                        Keycode::Num2 => keypad[0x2] = false,
                        Keycode::Num3 => keypad[0x3] = false,
                        Keycode::Num4 => keypad[0xC] = false,
                        Keycode::Q => keypad[0x4] = false,
                        Keycode::W => keypad[0x5] = false,
                        Keycode::E => keypad[0x6] = false,
                        Keycode::R => keypad[0xD] = false,
                        Keycode::A => keypad[0x7] = false,
                        Keycode::S => keypad[0x8] = false,
                        Keycode::D => keypad[0x9] = false,
                        Keycode::F => keypad[0xE] = false,
                        Keycode::Z => keypad[0xA] = false,
                        Keycode::X => keypad[0x0] = false,
                        Keycode::C => keypad[0xB] = false,
                        Keycode::V => keypad[0xF] = false,
                        _ => {()},
                    }

                },
                _ => {continue;}
            }
        }

        // bitshifting AB to the left, and grabbing CD
        let op: u16 = (memory[pc as usize] as u16) << 8 | memory[(pc as usize) + 1] as u16;
        let nnn: u16 = (op & 0x0FFF).try_into().unwrap();
        let n: u8 = (op & 0x000F).try_into().unwrap();
        let x: u8 = ((op & 0x0F00) >> 8).try_into().unwrap();
        let y: u8 = ((op & 0x00F0) >> 4).try_into().unwrap();
        let kk: u8 = (op & 0x00FF).try_into().unwrap();
        // nnn or addr - A 12-bit value, the lowest 12 bits of the instruction
        // n or nibble - A 4-bit value, the lowest 4 bits of the instruction
        // x - A 4-bit value, the lower 4 bits of the high byte of the instruction
        // y - A 4-bit value, the upper 4 bits of the low byte of the instruction
        // kk or byte - An 8-bit value, the lowest 8 bits of the instruction

        // matching the opcode to the first significant digit ()
        match op & 0xF000 {
            0x0000 => {
                match nnn & 0xFFF {
                    0x0E0 => { // Clearing the display
                        display = [[0; 64]; 32];
                        pc += 2;
                    },

                    0x0EE => { // Return from subroutine
                        pc = stack[sp as usize];
                        sp -= 1;
                    },
                    
                    _ => { // 0x0nnn, not used 
                        unreachable!("Unexpected opcode - 0x0nnn unused: {:04X}", op)
                    },     
                }
            },

            0x1000 => {
                pc = nnn; // Set the program counter
            },

            0x2000 => { // Calls a subroutine, sets the pc to nnn
                if usize::from(sp) >= stack.len() - 1 {
                    return Err(anyhow!("Stack overflow: subroutine depth exceeded"))
                } else {
                    sp += 1;
                    stack[sp as usize] = pc + 2;
                    pc = nnn;
                }
            },
            0x3000 => {
                if vx[x as usize] == kk { // Skip next instruction if Vx = kk.
                    pc += 2;
                }
                pc+=2;
            },
            0x4000 => {
                if vx[x as usize] != kk { // Skip next instruction if Vx != kk.
                    pc += 2;
                }
                pc+=2;
            },
            0x5000 => {
                if vx[x as usize] == vx[y as usize] { // Skip next instruction if Vx = Vy.
                    pc += 2;
                }
                pc+=2;
            },
            0x6000 => { //  Set Vx = kk.
                vx[x as usize] = kk;
                pc += 2;
            },
            0x7000 => { // Set Vx = Vx + kk.
                vx[x as usize] = vx[x as usize].wrapping_add(kk);
                pc += 2;
            },
            0x8000 => { // n = opcode & 0x000F
                match n   {
                    0x0 => { //  Set Vx = Vy.
                        vx[x as usize] = vx[y as usize];
                        pc += 2;
                    },
                    0x1 => { //  Set Vx = Vx OR Vy.
                        vx[x as usize] = vx[x as usize] | vx[y as usize]; 
                        pc += 2;
                    },
                    0x2 => { //  Set Vx = Vx AND Vy.
                        vx[x as usize] = vx[x as usize] & vx[y as usize];
                        pc += 2;
                    },
                    0x3 => { //  Set Vx = Vx XOR Vy.
                        vx[x as usize] = vx[x as usize] ^ vx[y as usize];
                        pc += 2;
                    },

                    0x4 => { //  Set Vx = Vx + Vy, set VF = carry.
                        let sum = vx[x as usize] as u16 + vx[y as usize] as u16;
                        vx[0xF] = if sum > 0xFF { 1 } else { 0 };
                        vx[x as usize] = sum as u8;
                        pc += 2;
                    },
                    0x5 => { //  If Vx > Vy, then VF is set to 1, otherwise 0. Then Vy is subtracted from Vx, and the results stored in Vx.
                        vx[0xF] = if vx[x as usize] > vx[y as usize] { 1 } else { 0 };
                        vx[x as usize] = vx[x as usize].wrapping_sub(vx[y as usize]);
                        pc += 2;
                    },
                    0x6 => { //  If the least-significant bit of Vx is 1, then VF is set to 1, otherwise 0. Then Vx is divided by 2.
                        vx[0xF] = vx[x as usize] & 0x1;
                        vx[x as usize] = vx[x as usize] >> 1;
                        pc += 2;
                    },
                    0x7 => { //  If Vy > Vx, then VF is set to 1, otherwise 0. Then Vx is subtracted from Vy, and the results stored in Vx.
                        vx[0xF] = if vx[y as usize] > vx[x as usize] { 1 } else { 0 };
                        vx[x as usize] = vx[y as usize].wrapping_sub(vx[x as usize]);
                        pc += 2; 
                    },
                    0xE => { //  If the most-significant bit of Vx is 1, then VF is set to 1, otherwise to 0. Then Vx is multiplied by 2.
                        vx[0xF] = (vx[x as usize] >> 7) & 0x1; // Shifts right by 7 bits and takes &
                        vx[x as usize] = vx[x as usize] << 1; // shifts left 1
                        pc += 2;
                    },
                    _ => {panic!("Unknown opcode: 0x{:X}", op)},
                }
            },
            0x9000 => { // Skip next instruction if Vx != Vy.
                if (op & 0x000F) == 0 {
                    pc += if vx[x as usize] != vx[y as usize] { 4 } else { 2 };
                } else {
                    panic!("Unknown opcode: 0x{:X}", op);
                }
            },
            0xA000 => { //  Set I = nnn.
                i = nnn;
                pc+=2;
            },
            0xB000 => { //  Jump to location nnn + V0.
                pc = (vx[0x0] as u16) + nnn;
            },
            0xC000 => { //  Set Vx = random byte AND kk.
                let n1: u8 = rng.gen();
                vx[x as usize] = n1 & kk;
                pc += 2;
            },
            0xD000 => { //  Display n-byte sprite starting at memory location I at (Vx, Vy) [col, row], set VF = collision.
                // Do I really want to overwrite x, y here? Check this
                let x = vx[x as usize] as usize;
                let y = vx[y as usize] as usize;

                for row in 0..n{
                    let sbyte = memory[(i+row as u16) as usize];
                    let drow = (y + row as usize) % 32;

                    for col in 0..8{
                        let sp = (sbyte >> (7 - col) ) & 1; // bit shift to the right to identify if this pixel is 0 or 1 from the byte representation
                        let dcol = (x + col as usize) % 64;
                        if sp == 1 {
                            if display[drow][dcol] == 1 {
                                vx[0xF] = 1;
                            }
                            display[drow][dcol] ^= 1;
                        }
                    }
                }
                pc+=2;
            },
            0xE000 => {
                match kk {
                    0x9E => { // Skip next instruction if key with the value of Vx is pressed.
                        if keypad[vx[x as usize] as usize] {
                            pc +=4;
                        } else { pc += 2 }
                    },
                    0xA1 => { // Skip next instruction if key with the value of Vx is not pressed.
                        if !keypad[vx[x as usize] as usize] {
                            pc += 4;
                        } else {pc += 2; }
                    },
                    _ => {panic!("Unknown opcode: 0x{:X}", op)},
                }
            },

            0xF000 => {
                match kk {
                    0x07 => { //  Set Vx = delay timer value.
                        vx[x as usize] = delay;
                        pc+=2;
                    },
                    0x0A => { //  Wait for a key press, store the value of the key in Vx.
                        let mut waiting = true;
                        while waiting {
                            for key in 0..16 {
                                if keypad[key] == true{
                                    vx[x as usize] = key as u8;
                                    waiting = false;
                                }
                            }
                        }
                        pc+=2;
                    },
                    0x15 => { // Set delay timer = Vx.
                        delay = vx[x as usize];
                        pc+=2;
                    },
                    0x18 => { // Set sound timer = Vx.
                        sound = vx[x as usize];
                        pc+=2;                    
                    },
                    0x1E => { //  Set I = I + Vx.
                        i += vx[x as usize] as u16;
                        pc+=2;
                    },
                    0x29 => { //  Set I = location of sprite for digit Vx.
                        i = (5 * vx[x as usize]) as u16;
                        pc+=2;
                    },
                    0x33 => { // Store BCD representation of Vx in memory locations I, I+1, and I+2.
                        let hund = (((vx[x as usize] as u16) % 1000) / 100) as u8;
                        let tens = (vx[x as usize] % 100) / 10;
                        let ones = vx[x as usize] % 10; // lol
                        memory[i as usize] = hund;
                        memory[(i as usize)+1] = tens;
                        memory[(i as usize)+2] = ones;
                        pc+=2;
                    }
                    0x55 => { // Store registers V0 through Vx in memory starting at location I.
                        for idx in 0..=x {
                            memory[(i as usize) + (idx as usize)] = vx[idx as usize];
                        }
                        i += (x + 1) as u16; 
                        pc += 2;
                    },
                    0x65 => { //  Read registers V0 through Vx from memory starting at location I.
                        for idx in 0..=x {
                            vx[idx as usize] = memory[(i as usize) + (idx as usize)]
                        }
                        pc +=2;
                    }, 
                    _ => todo!()
                }
            },
            _ => {unreachable!("Unexpected opcode: {:04X}", op)},
        }

        // Playing a sound if needed
        if sound > 0 && sink.lock().unwrap().empty() {
            play_sound(sink.clone(), source.clone(), sin_delay.clone());
        }
        // Time update here
        let now = std::time::Instant::now();
        let elapsed = now.duration_since(last_timer_update);
        if elapsed >= Duration::from_nanos(16_666_667) {
            if delay > 0 {
                delay -= 1;
            } 
            if sound > 0 {
                sound -= 1;
            }
            last_timer_update = now;

        }
        // Handle display here
        render_display(&mut canvas, &display, scale)?;
        thread::sleep(Duration::from_millis(1000 / refresh as u64)); // Refresh rate
    }
    Ok(())
}
