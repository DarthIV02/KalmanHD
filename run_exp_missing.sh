for noise in 0.1 0.2 0.4 0.5
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
        
        python3 main.py --model RegHD --dataset "$dataset" --learning_rate "$learning_rate" --hd_representation "$hd_representation" --p "$noise";

        # python3 main.py --model KalmanFilter --dataset "$dataset" --p "$noise";
        
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

        # python3 main.py --dataset "$dataset" --learning_rate "$learning_rate" --hd_representation "$hd_representation" --p "$noise";
    done
done