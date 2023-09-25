import subprocess

for noise in [0.1, 0.2, 0.5, 1]:
    for dataset in ["SanFranciscoTraffic"]: #"MetroInterstateTrafficVolume", "GuangzhouTraffic", "EnergyConsumptionFraunhofer", "ElectricityLoadDiagrams"
        # subprocess.call(f"python main.py --model VAE --dataset {dataset} --gaussian_noise {noise}", shell=True)

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
        
        #subprocess.call(f"python main.py --model RegHD --learning_rate {learning_rate} --hd_representation {hd_representation} --dimension_hd {dimension_hd} --dataset {dataset} --gaussian_noise {noise}", shell=True)

    #subprocess.call(f"python main.py --gaussian_noise {noise}", shell=True)

    #subprocess.call(f"python main.py --gaussian_noise {noise} --models 1", shell=True)
    
    subprocess.call(f"python main.py --gaussian_noise {noise} --models 4 --novelty 0.2", shell=True)
    subprocess.call(f"python main.py --gaussian_noise {noise} --models 4 --novelty 0.4", shell=True)
    subprocess.call(f"python main.py --gaussian_noise {noise} --models 4 --novelty 0.6", shell=True)
    subprocess.call(f"python main.py --gaussian_noise {noise} --models 4 --novelty 0.8", shell=True)
    subprocess.call(f"python main.py --gaussian_noise {noise} --models 8 --novelty 0.2", shell=True)
    subprocess.call(f"python main.py --gaussian_noise {noise} --models 8 --novelty 0.4", shell=True)
    subprocess.call(f"python main.py --gaussian_noise {noise} --models 8 --novelty 0.6", shell=True)
    subprocess.call(f"python main.py --gaussian_noise {noise} --models 8 --novelty 0.8", shell=True)