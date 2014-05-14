fprintf(_filenameStore, "local time: %02d-%02d-%02d  %2d:%02d\n",ptm->tm_mday,ptm->tm_mon+1,ptm->tm_year+1900, (ptm->tm_hour+2)%24, ptm->tm_min);       
			fprintf(_filenameStore, "Outcome: %i\n", OutcomeCondiviso);
			for (cycletime_i=0;cycletime_i<cyclesToPrint;cycletime_i++)
			{
				fprintf(_filenameStore, "%i %i %u ", cycletime[cycletime_i],cycleproduced[cycletime_i],cyclebuffer[cycletime_i]);
				fprintf(_filenameStore, "%i %i %i ", walkInfo[cycletime_i],opInfo[cycletime_i],contrInfo[cycletime_i]);
				fprintf(_filenameStore, "%f ", kinDebugInfo[cycletime_i]);
				for (j=0;j<BRbinN*CellsPerBin; j++)
					fprintf(_filenameStore, "%f ", binsgot[cycletime_i][j]);
				fprintf(_filenameStore, "; \n");
			}

		cycletime[cycletime_i] = m_rz5.GetTargetVal(tdt_cycletime);
		cyclebuffer[cycletime_i] = timeflop;
		cycleproduced[cycletime_i] = m_rz5.GetTargetVal(tdt_cycleproduced);	
		
		for (j=0;j<BRbinN*CellsPerBin; j++)
			binsgot[cycletime_i][j] = BRcellVal[j];
		walkInfo[cycletime_i] = ratState;
		opInfo[cycletime_i] = OpState;
		contrInfo[cycletime_i] = globContrH;

		kinDebugInfo[cycletime_i] = GenCurFrameNumber;
		cycletime_i++;

opInfo
	OpState=0;
	OpStateNull=0;
	OpStateError=1;
	OpStateRear=2;
	OpStateReward=3;


BEFORE SCI (HEALTHY) = observation
ratState
	 ratStateNull=0;
	 ratStateStop=1;
	 ratStateWalk=2;
	 ratStateRear=3;

AFTER SCI (SCI) = guess by GMM
walkInfo
	brainState0 = -4.0;			//rest
	brainState1 = -2.0;			//Init
	brainState2 = 0.0;			//stop
	brainState3 = 2.0;			//walk
	brainState3 = 4.0;			//rear




cycletime -> how many ms passed from new data available to end of cpp computations (measured by TDT)
cycleproduced -> number of cycle from tdt
cyclebuffer -> similar to cycleproduced but increments of 32 each cycle


globContrH / contrInfo
-1 -> before the rat is positioned (still) (only SCI)
0 -> before initiation
1 -> Initiation sound played, rat not moving
2 -> rat walking
3 -> stop sound played, rat still walking
4 -> after end of trial (extra recordings)

time elapsed	cycleCount		cycleCount*32	WalkInfo	opInfo		contrInfo	Kindebug
65				4910			544				1			0			0			0.000000
