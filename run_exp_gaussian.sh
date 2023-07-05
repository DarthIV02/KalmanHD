for noise in 0.1 0.2 0.5 1
do
    for dataset in SanFranciscoTraffic MetroInterstateTrafficVolume GuangzhouTraffic EnergyConsumptionFraunhofer ElectricityLoadDiagrams
    do
        python3 main.py --learning_rate 0.00001 --dimension_hd 10000 --model RegHD --dataset "$dataset" --gaussian_noise "$noise";

        python3 main.py --model KalmanFilter --dataset "$dataset" --gaussian_noise "$noise";
        
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

        python3 main.py --dataset "$dataset" --learning_rate "$learning_rate" --hd_representation "$hd_representation" --gaussian_noise "$noise";
    done
done