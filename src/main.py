import case_control as cc
import argparse


def get_args():
    parser = argparse.ArgumentParser(description='Tutorial workflow for AWF.')
    parser.add_argument("-i", "--inputFile", help="path to input json file")
    parser.add_argument("-o", "--outputDir", help="path to output directory")
    args = parser.parse_args()
    return args


def main():
    """Method contains the main program flow. Refer to TGN for more guidance."""

    print('*** SOME - NAME - OF - THE - TOOL 0.1 beta ***')
    print('SFE tool for probabilistic design fire generation and  quantified risk assessment\n')
    print(
        'Special thanks to UKMEA Fire team, AWF and TDA dev teams, as well as Arup Digital Strategic Development Fund.')
    print('Contact: Yavor Panev at yavor.panev@arup.com\n')
    print('*** Analysis Start ***\n')

    args = get_args()

    # Method fetches all inputs and configurations. Sets up the folder space
    analysis = cc.CaseControler(inputs=args.inputFile, out_f=args.outputDir)
    analysis.initiate_methods()

    # This includes: 1) Calculation of protection requirements for different standard exposure
    # 2) Assessment of risk target
    analysis.conduct_pre_run_calculations()
    # Analysis of all cases according to run a setup parameters
    analysis.run_a_study()
    # Analysis of max, min, and median case according to run b setup parameters
    analysis.run_b_study()

    print('*** Analysis Completed Successfully ***\n')


if __name__ == "__main__":
    main()
