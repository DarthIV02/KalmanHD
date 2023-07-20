for models in 2 8 32
do
    for dataset in SanFranciscoTraffic MetroInterstateTrafficVolume GuangzhouTraffic EnergyConsumptionFraunhofer ElectricityLoadDiagrams
    do
        if [ "$dataset" = "SanFranciscoTraffic" ]; then
            learning_rate=0.01
            hd_representation=4
            for novelty in 0.003 0.005 0.006
            do
                python3 main.py --model KalmanHD --models "$models" --novelty "$novelty" --dataset "$dataset" --learning_rate "$learning_rate" --hd_representation "$hd_representation";
            done
            
        elif [ "$dataset" = "MetroInterstateTrafficVolume" ]; then
            learning_rate=0.1
            hd_representation=2
            for novelty in 0.02 0.04 0.05
            do
                python3 main.py --model KalmanHD --models "$models" --novelty "$novelty" --dataset "$dataset" --learning_rate "$learning_rate" --hd_representation "$hd_representation";
            done
            
        elif [ "$dataset" = "GuangzhouTraffic" ]; then
            learning_rate=0.01
            hd_representation=4
            for novelty in 0.002 0.003 0.004
            do
                python3 main.py --model KalmanHD --models "$models" --novelty "$novelty" --dataset "$dataset" --learning_rate "$learning_rate" --hd_representation "$hd_representation";
            done

        elif [ "$dataset" = "EnergyConsumptionFraunhofer" ]; then
            learning_rate=0.01
            hd_representation=4
            for novelty in 0.002 0.004 0.005
            do
                python3 main.py --model KalmanHD --models "$models" --novelty "$novelty" --dataset "$dataset" --learning_rate "$learning_rate" --hd_representation "$hd_representation";
            done

        elif [ "$dataset" = "ElectricityLoadDiagrams" ]; then
            learning_rate=0.01
            hd_representation=1
            for novelty in 0.70 0.75 0.80
            do
                python3 main.py --model KalmanHD --models "$models" --novelty "$novelty" --dataset "$dataset" --learning_rate "$learning_rate" --hd_representation "$hd_representation";
            done
        fi
    done
done