use std::io;
use std::env;
use std::process;
use std::fs;
use std::fs::File;
use std::io::Write;
use image::io::Reader as ImageReader;
use image::{ImageError, GenericImageView, DynamicImage, Pixel};

fn load_image(filepath: String) -> Result<image::DynamicImage, ImageError> {
    let image = ImageReader::open(filepath)?.decode()?;
    Ok(image)
}
fn img_to_ascii(img: image::DynamicImage, scaling: f32) -> Vec<String> {

    let img_dim: (u32, u32) = img.dimensions();
    let width = img_dim.0;
    let height = img_dim.1;
    let nwidth = (width as f32 * scaling) as u32;
    let nheight = (height as f32 * scaling) as u32;
    let resized_img = img.resize(nwidth, nheight, image::imageops::FilterType::Lanczos3);
    let grayscale = resized_img.into_luma8();

    let mut ascii_out: Vec<String> = Vec::new();
    let chars_ = [' ', '.', ':', '-', '=', '+', '*', '#', '%', '@'];

    for y in 0..grayscale.height() {
        let mut row_string: String = "".to_string();
        for x in 0..grayscale.width() {
            // Accessing the pixel and checking its darkness in the 0 channel (already grayscale)
            let color = grayscale.get_pixel(x, y).0[0];
            let char_i = ((color as f32 / 255.0) * (chars_.len() - 1) as f32).round() as usize;
            row_string.push(chars_[char_i]);
            row_string.push(' ');
        }
        ascii_out.push(row_string);
    } 
   

    println!("Image dimensions: {} x {}", width, height);
    println!("Image dimensiosn after rescale: {} x {}", nwidth, nheight);
    ascii_out
}
fn main()-> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = env::args().skip(1).collect();
    if args.is_empty() {
        println!("Call with filename as an argument. Include extension. \n Images are stored in /images/");
        process::exit(1);
    }
    // Checking if the file exists
    let filen: String = args[0].to_owned();
    let mut scaling = 1.0;
 
    println!("debugging: args are {:?}", args);
    
    if args.len() > 1 {
        
        let scaling_str = &args[1];
        scaling = match scaling_str.parse::<f32>() {
            Ok(value) => value,
            Err(_) => {
                eprintln!("Second arg must be a float");
                std::process::exit(1);
            }
        };
    } 

    let path: String = "images/".to_owned();
    let filepath = format!("{path}{filen}");
    let _metadata = match fs::metadata(&filepath) {
        Ok(_metadata) => {
            if _metadata.is_file() {
                println!("File {} being loaded", filepath);
                let img = load_image(filepath)?;
                let asciiimg = img_to_ascii(img, scaling);    
                // Printing and sending to txt
                let ascit = asciiimg.iter();
                for row in ascit {
                    println!("{:?}", row);
                }

                let mut file = match File::create("ascii.txt") {
                    Ok(file) => file,
                    Err(e) => {
                        eprintln!("Failed to create file: {}", e);
                        std::process::exit(1);
                    }
                };
                let ascit = asciiimg.iter();
                for row in ascit {
                    match file.write_all(row.as_bytes()) {
                        Ok(_) => {},
                        Err(e) => {
                            eprintln!("Failed to write file: {}{}", row, e);
                            std::process::exit(1);
                        }
                    }
                    match file.write_all(b"\n") {
                        Ok(_) => {},
                        Err(e) => {
                            eprintln!("Failed to write nl: {}", e);
                            std::process::exit(1);
                        }
                    }
                }                
            } else {
                println!("Filepath exists, file does not exists");
                process::exit(1);
            }
        }
        Err(err) => {
            eprintln!("Error: {}", err);
            process::exit(1);   
        }
    };
    // Loading the image if it exists
    //
    Ok(())
}
