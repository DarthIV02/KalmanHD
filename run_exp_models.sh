for models in 2 #8 32
do
    for dataset in SanFranciscoTraffic #MetroInterstateTrafficVolume GuangzhouTraffic EnergyConsumptionFraunhofer ElectricityLoadDiagrams
    do
        if [ "$dataset" = "SanFranciscoTraffic" ]; then
            learning_rate=0.01
            hd_representation=4
            for novelty in 0.3 0.4 0.5 0.6 0.7 0.8 0.9
            do
                python3 main.py --model KalmanHD --models "$models" --novelty "$novelty" --dataset "$dataset" --learning_rate "$learning_rate" --hd_representation "$hd_representation";
            done
            
        elif [ "$dataset" = "MetroInterstateTrafficVolume" ]; then
            learning_rate=0.1
            hd_representation=2
            #for novelty in 0.01 0.02 0.025
            #do
            #    python3 main.py --model KalmanHD --models "$models" --novelty "$novelty" --dataset "$dataset" --learning_rate "$learning_rate" --hd_representation "$hd_representation";
            #done
            
        elif [ "$dataset" = "GuangzhouTraffic" ]; then
            learning_rate=0.01
            hd_representation=4
            #for novelty in 0.001 0.0015 0.002
            #do
                #python3 main.py --model KalmanHD --models "$models" --novelty "$novelty" --dataset "$dataset" --learning_rate "$learning_rate" --hd_representation "$hd_representation";
            #done

        elif [ "$dataset" = "EnergyConsumptionFraunhofer" ]; then
            learning_rate=0.01
            hd_representation=4
            #for novelty in 0.001 0.002 0.0025
            #do
                #python3 main.py --model KalmanHD --models "$models" --novelty "$novelty" --dataset "$dataset" --learning_rate "$learning_rate" --hd_representation "$hd_representation";
            #done

        elif [ "$dataset" = "ElectricityLoadDiagrams" ]; then
            learning_rate=0.01
            hd_representation=1
            #for novelty in 0.30 0.35 0.40
            #do
                #python3 main.py --model KalmanHD --models "$models" --novelty "$novelty" --dataset "$dataset" --learning_rate "$learning_rate" --hd_representation "$hd_representation";
            #done
        fi
    done
done