use anyhow::Result;
use std::io::{self, Write};
use std::process::Command;
use talava_tracker::vmt::{TrackerPose, VmtClient};

/// WSL2環境でWindowsホストのIPを取得
fn get_wsl_host_ip() -> Option<String> {
    let output = Command::new("ip")
        .args(["route", "show", "default"])
        .output()
        .ok()?;

    let stdout = String::from_utf8_lossy(&output.stdout);
    // "default via 172.28.176.1 dev eth0 ..." から IPを抽出
    stdout.split_whitespace().nth(2).map(|s| s.to_string())
}

/// VMTアドレスを決定
fn get_vmt_addr() -> String {
    // 環境変数が設定されていればそれを使用
    if let Ok(addr) = std::env::var("VMT_ADDR") {
        return addr;
    }

    // WSL環境ならWindowsホストIPを使用
    if std::path::Path::new("/proc/sys/fs/binfmt_misc/WSLInterop").exists() {
        if let Some(host_ip) = get_wsl_host_ip() {
            return format!("{}:39570", host_ip);
        }
    }

    // デフォルトはlocalhost
    "127.0.0.1:39570".to_string()
}

fn main() -> Result<()> {
    println!("=== Talava Tracker - VMT Test ===");
    println!("VMTに接続します...");

    let vmt_addr = get_vmt_addr();
    let client = VmtClient::new(&vmt_addr)?;
    println!("接続先: {}", vmt_addr);
    println!();
    println!("コマンド:");
    println!("  p x y z       - 位置を設定 (例: p 0 1 0)");
    println!("  r x y z w     - 回転を設定 (例: r 0 0 0 1)");
    println!("  s             - 現在の値を送信");
    println!("  t             - テスト送信 (位置を少しずつ動かす)");
    println!("  q             - 終了");
    println!();

    let mut pose = TrackerPose::identity();
    let index = 0;
    let enable = true;

    loop {
        print!("> ");
        io::stdout().flush()?;

        let mut input = String::new();
        io::stdin().read_line(&mut input)?;
        let parts: Vec<&str> = input.trim().split_whitespace().collect();

        if parts.is_empty() {
            continue;
        }

        match parts[0] {
            "p" if parts.len() == 4 => {
                let x: f32 = parts[1].parse()?;
                let y: f32 = parts[2].parse()?;
                let z: f32 = parts[3].parse()?;
                pose.position = [x, y, z];
                println!("位置: [{}, {}, {}]", x, y, z);
                client.send(index, enable, &pose)?;
                println!("送信しました");
            }
            "r" if parts.len() == 5 => {
                let x: f32 = parts[1].parse()?;
                let y: f32 = parts[2].parse()?;
                let z: f32 = parts[3].parse()?;
                let w: f32 = parts[4].parse()?;
                pose.rotation = [x, y, z, w];
                println!("回転: [{}, {}, {}, {}]", x, y, z, w);
                client.send(index, enable, &pose)?;
                println!("送信しました");
            }
            "s" => {
                println!("現在の値:");
                println!("  位置: {:?}", pose.position);
                println!("  回転: {:?}", pose.rotation);
                client.send(index, enable, &pose)?;
                println!("送信しました");
            }
            "t" => {
                println!("テスト送信中...");
                for i in 0..10 {
                    let y = i as f32 * 0.1;
                    pose.position[1] = y;
                    client.send(index, enable, &pose)?;
                    println!("  y = {}", y);
                    std::thread::sleep(std::time::Duration::from_millis(500));
                }
                println!("テスト完了");
            }
            "q" => {
                println!("終了します");
                break;
            }
            _ => {
                println!("不明なコマンド: {}", parts[0]);
            }
        }
    }

    Ok(())
}
