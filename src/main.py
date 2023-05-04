import case_control as cc

print('*** SOME - NAME - OF - THE - TOOL 0.1 beta ***')
print('SFE tool for probabilistic design fire generation and  quantified risk assessment\n')
print('Special thanks to UKMEA Fire team, AWF and TDA dev teams, as well as Arup Digital Strategic Development Fund.')
print('Contact: Yavor Panev at yavor.panev@arup.com\n')
print('*** Analysis Start ***\n')

analysis = cc.CaseControler(inputs=r'samples\test_input.json', out_f=r'dump')
analysis.initiate_methods()
analysis.conduct_pre_run_calculations()
analysis.run_a_study()
analysis.run_b_study()

print('*** Analysis Completed Successfully ***\n')