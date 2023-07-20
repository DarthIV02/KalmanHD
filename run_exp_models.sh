for models in 2 8 32
do
    for novelty in 0.05 0.07 0.09 0.1 0.12 0.15
    do
        for dataset in SanFranciscoTraffic MetroInterstateTrafficVolume GuangzhouTraffic EnergyConsumptionFraunhofer ElectricityLoadDiagrams
        do
            if [ "$dataset" = "SanFranciscoTraffic" ]; then
                learning_rate=0.01
                hd_representation=4
                y=100
                novelty=$(bc <<< "scale=4;$novelty/$y")
            elif [ "$dataset" = "MetroInterstateTrafficVolume" ]; then
                learning_rate=0.1
                hd_representation=2
                y=10
                novelty=$(bc <<< "scale=4;$novelty/$y")
            elif [ "$dataset" = "GuangzhouTraffic" ]; then
                learning_rate=0.01
                hd_representation=4
                y=100
                novelty=$(bc <<< "scale=4;$novelty/$y")
            elif [ "$dataset" = "EnergyConsumptionFraunhofer" ]; then
                learning_rate=0.01
                hd_representation=4
                y=100
                novelty=$(bc <<< "scale=4;$novelty/$y")
            elif [ "$dataset" = "ElectricityLoadDiagrams" ]; then
                learning_rate=0.01
                hd_representation=1
                y=1
            fi
            
            python3 main.py --model KalmanHD --models "$models" --novelty "$novelty" --dataset "$dataset" --learning_rate "$learning_rate" --hd_representation "$hd_representation";

            novelty=$(bc <<< "scale=4;$novelty*$y")
        done
    done
done