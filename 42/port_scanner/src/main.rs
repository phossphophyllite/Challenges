use std::env;
use std::net::TcpStream;
use std::str::FromStr;
//use std::thread;
use std::time::Duration;


fn main() {
    let args: Vec<String> = env::args().collect();
    if args.len() < 4 {
        eprintln!("Wrong args - usage is {} <ip> <start_port> <end_port>", args[0]);
        return;
    }
    let ip = &args[1];
    let start_port = u16::from_str(&args[2]).expect("Invalid start port");
    let end_port = u16::from_str(&args[3]).expect("Invalid end port");
    let is_ipv6 = ip.contains(":");

    for port in start_port..=end_port {
        let address = if is_ipv6 {
            format!("[{}]:{}", ip, port)
        }
        else {
            format!("{}:{}", ip, port)
        };

               if let Ok(_) = TcpStream::connect_timeout(&address.parse().unwrap(), Duration::from_secs(1)) {
            println!("Port {} is open", port);
        }
    }
    
}
