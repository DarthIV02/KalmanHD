import subprocess

for noise in [0.01, 0.02, 0.05]:
    for dataset in ["SanFranciscoTraffic", "MetroInterstateTrafficVolume", "GuangzhouTraffic", "EnergyConsumptionFraunhofer", "ElectricityLoadDiagrams"]:
        subprocess.call(f"python main.py --model DNN --dataset {dataset} --flipping_rate {noise}", shell=True)