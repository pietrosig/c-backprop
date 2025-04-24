import subprocess
import torch

def run_make_test():
    print("🛠️  Running `make test`...")
    result = subprocess.run(["make", "test-numeric"], capture_output=True, text=True)
    if result.returncode != 0:
        print("❌ Make failed:")
        print(result.stderr)
        return False
    print("✅ Make succeeded")
    return True

def run_test_binary(binary_path="./test_numeric_bin"):
    print(f"🚀 Running test binary `{binary_path}`...")
    result = subprocess.run([binary_path], capture_output=True, text=True)
    if result.returncode != 0:
        print("❌ Test binary exited with errors:")
        print(result.stderr)
    else:
        print("✅ Test binary ran successfully.")
    output = result.stdout
    output.split("\n")

    out_dict = {}
    for line in output.split("\n"):
        if line:
            key, value = line.split(":")
            out_dict[key.strip()] = value.strip()

    return out_dict

def run_pytorch_test():
    print(f"🚀 Running PyTorch test...")

    a = torch.tensor(-3.6, dtype=torch.float, requires_grad=True)
    b = torch.tensor(2.312, dtype=torch.float, requires_grad=True)
    c = torch.tensor(7.95, dtype=torch.float, requires_grad=True)
    d = torch.tensor(-271, dtype=torch.float, requires_grad=True)
    e = torch.tensor(-932.229, dtype=torch.float, requires_grad=True)

    # (ReLu(cos(a - b) / c) * d) - (sin(a / c) / e)
    loss =  (torch.nn.functional.relu(torch.cos(a - b) / c) * d) - (torch.sin(a / c) / e)

    loss.backward()

    out_dict = {}
    out_dict["loss"] = "{:.6f}".format(loss.item())
    out_dict["a_grad"] = "{:.6f}".format(a.grad.item())
    out_dict["b_grad"] = "{:.6f}".format(b.grad.item())
    out_dict["c_grad"] = "{:.6f}".format(c.grad.item())
    out_dict["d_grad"] = "{:.6f}".format(d.grad.item())
    out_dict["e_grad"] = "{:.6f}".format(e.grad.item())

    print("✅ Done.")

    return out_dict

if __name__ == "__main__":
    if run_make_test():
        c_out = run_test_binary()
        torch_out = run_pytorch_test()

        TOLERANCE = 1e-5
        print(f"🔍 Comparing outputs with tollerance of {TOLERANCE}...")
        for key in torch_out.keys():
            if key in c_out:
                try:
                    a = float(torch_out[key])
                    b = float(c_out[key])
                    if abs(a - b) > TOLERANCE:
                        print(f"❌ {key} mismatch: PyTorch: {a:.6f}, C: {b:.6f} (Δ={abs(a - b):.2e})")
                    else:
                        print(f"✅ {key} match: {a:.7f}")
                except ValueError:
                    # fallback if values aren't floats
                    if torch_out[key] != c_out[key]:
                        print(f"❌ {key} mismatch (non-float): PyTorch: {torch_out[key]}, C: {c_out[key]}")
                    else:
                        print(f"✅ {key} match: {torch_out[key]}")
            else:
                print(f"⚠️  {key} not found in C output")

