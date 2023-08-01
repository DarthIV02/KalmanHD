import subprocess

for noise in [0.1, 0.2, 0.4, 0.5]:
    for dataset in ["SanFranciscoTraffic"]: #"MetroInterstateTrafficVolume", "GuangzhouTraffic", "EnergyConsumptionFraunhofer", "ElectricityLoadDiagrams"
        #subprocess.call(f"python main.py --model VAE --dataset {dataset} --p {noise}", shell=True)

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

        print(noise)
        
        subprocess.call(f"python main.py --p {noise} --models 2", shell=True)

        subprocess.call(f"python main.py --p {noise} --models 1", shell=True)