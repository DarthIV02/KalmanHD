for dataset in SanFranciscoTraffic MetroInterstateTrafficVolume GuangzhouTraffic EnergyConsumptionFraunhofer ElectricityLoadDiagrams
do

    python3 main.py --model KalmanFilter --dataset "$dataset";
    
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

    python3 main.py --model KalmanHD --dataset "$dataset" --learning_rate "$learning_rate" --hd_representation "$hd_representation";

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
    
    python3 main.py --learning_rate "$learning_rate" --dimension_hd "$dimension_hd" --model RegHD --dataset "$dataset" --hd_representation "$hd_representation";

    python3 main.py --model DNN --dataset "$dataset";

    python3 main.py --model VAE --dataset "$dataset";

done
