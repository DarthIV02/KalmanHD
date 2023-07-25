import subprocess

for dataset in ["SanFranciscoTraffic", "MetroInterstateTrafficVolume", "GuangzhouTraffic", "EnergyConsumptionFraunhofer", "ElectricityLoadDiagrams"]:
        if dataset == "SanFranciscoTraffic":
            learning_rate=0.000001
            hd_representation=4
            dimension_hd=2000
        elif dataset == "MetroInterstateTrafficVolume":
            learning_rate=0.00001
            hd_representation=2
            dimension_hd=1000
        elif dataset == "GuangzhouTraffic":
            learning_rate=0.000001
            hd_representation=4
            dimension_hd=1000
        elif dataset == "EnergyConsumptionFraunhofer":
            learning_rate=0.000001
            hd_representation=4
            dimension_hd=1000
        elif dataset == "ElectricityLoadDiagrams":
            learning_rate=0.00001
            hd_representation=1
            dimension_hd=2000

        #for dimension_hd in [1000, 2000, 5000, 10000]:
        subprocess.call(f"python main.py --model RegHD --learning_rate {learning_rate} --hd_representation {hd_representation} --dimension_hd {dimension_hd} --dataset {dataset}", shell=True)