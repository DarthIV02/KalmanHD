for noise in 0.01 0.02 0.05 0.10 0.15
do
    for dataset in SanFranciscoTraffic MetroInterstateTrafficVolume GuangzhouTraffic EnergyConsumptionFraunhofer ElectricityLoadDiagrams
    do
        if [ "$dataset" = "SanFranciscoTraffic" ]; then
            learning_rate=0.000001
            hd_representation=4
            dimension_hd=2000
        elif [ "$dataset" = "MetroInterstateTrafficVolume" ]; then
            learning_rate=0.00001
            hd_representation=2
            dimension_hd=1000
        elif [ "$dataset" = "GuangzhouTraffic" ]; then
            learning_rate=0.000001
            hd_representation=4
            dimension_hd=2000
        elif [ "$dataset" = "EnergyConsumptionFraunhofer" ]; then
            learning_rate=0.000001
            hd_representation=4
            dimension_hd=2000
        elif [ "$dataset" = "ElectricityLoadDiagrams" ]; then
            learning_rate=0.00001
            hd_representation=1
            dimension_hd=5000
        fi
        
        python3 main.py --model RegHD --dataset "$dataset" --learning_rate "$learning_rate" --hd_representation "$hd_representation" --dimension_hd "$dimension_hd" --flipping_rate "$noise";

        # python3 main.py --model KalmanFilter --dataset "$dataset" --gaussian_noise "$noise";
        
        if [ "$dataset" = "SanFranciscoTraffic" ]; then
            learning_rate=0.01
            hd_representation=4
        elif [ "$dataset" = "MetroInterstateTrafficVolume" ]; then
            learning_rate=0.1
            hd_representation=2
        elif [ "$dataset" = "GuangzhouTraffic" ]; then
            learning_rate=0.01
            hd_representation=4
        elif [ "$dataset" = "EnergyConsumptionFraunhofer" ]; then
            learning_rate=0.01
            hd_representation=4
        elif [ "$dataset" = "ElectricityLoadDiagrams" ]; then
            learning_rate=0.01
            hd_representation=1
        fi

        python3 main.py --model KalmanHD --dataset "$dataset" --learning_rate "$learning_rate" --hd_representation "$hd_representation" --flipping_rate "$noise";

        # python3 main.py --model DNN --dataset "$dataset" --gaussian_noise "$noise";
    done
done