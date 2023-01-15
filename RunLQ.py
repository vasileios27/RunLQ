from __future__ import division
from __future__ import print_function

import os,sys,dill
import numpy as np
from scipy.integrate import solve_ivp, OdeSolution
"""
The lines below make available the path of the WorkOn folder
"""
cluster = False
if not cluster:
    Path_to_WorkOn = '/Users/vasileios_vatellis/Desktop/WorkOn'
    Path_to_TestLQDRpy = '/Users/vasileios_vatellis/Desktop/WorkOn/DRalgoDev/output'
    if os.path.abspath(Path_to_WorkOn) not in sys.path:
        sys.path.append(Path_to_WorkOn)
    if os.path.abspath(Path_to_TestLQDRpy) not in sys.path:
        sys.path.append(Path_to_TestLQDRpy)
if cluster:
    Path_to_WorkOn = '/home/vasileios.vatellis@UA.PT/WorkOn'
    Path_to_TestLQDRpy = '/home/vasileios.vatellis@UA.PT/WorkOn/DRalgoDev/output'
    if os.path.abspath(Path_to_WorkOn) not in sys.path:
        sys.path.append(Path_to_WorkOn)
    if os.path.abspath(Path_to_TestLQDRpy) not in sys.path:
        sys.path.append(Path_to_TestLQDRpy)

import TestLQDRpy as DR
from CosmoTransitions.CosmoTransitions import generic_potential
from CosmoTransitions.CosmoTransitions import transitionFinder

"""
The next line will create the output folder where we are going to save the data
"""
fol = False
wn = 0
while not fol:
    wn +=1
    if "LQbatch_{}".format(wn) not in os.listdir():
        fol=True
        outputFol = os.path.join(os.getcwd(),'LQbatch_{}'.format(wn))
        os.makedirs(outputFol)
outputFol = os.path.join(os.getcwd(),'LQbatch_{}'.format(wn))

global Pi
Pi = np.pi

Sqrt = np.sqrt

class gaugeCouplingsTreeLeve():
    """
    The goal of this class is to calculate the tree levele couplings of the
    Leptoquark model
    CosmoTransition class
    """
    def __init__(self, Lowmass=1.5e03, Highmass=4.0e03, Lowcoup=0.0, Highcoup=12,\
        Lowcubic=0.0, Highcubic=400.0 ):
        """
        In this fuction we initialize the values
        We give random masses for 2 of the LQ particles
        Lowmass: low limit of the mass
        Highmass: upper limit of the mass

        We also give random values to the "BSM couplings"
        Lowcoup: lower value that can get
        Highcoup: the highest value they can get
        """
        self.Lowmass, self.Highmass, self.Lowcoup, self.Highcoup, self.Lowcubic, \
        self.Highcubic = Lowmass, Highmass, Lowcoup, Highcoup, Lowcubic, Highcubic

        self.ghs, self.ghr, self.grs,self.ghr2  = np.random.uniform(low=Lowcoup,high=Highcoup,size=4)
        self.λr, self.λs =  np.random.uniform(low=0,high=Highcoup,size=2)

        self.a1 = np.random.uniform(low=Lowcubic,high=Highcubic)
        self.mR1sq,self.mSsq = np.random.uniform(low=Lowmass**2,high=Highmass**2,size = 2)
        self.mSsq,self.mR1sq = np.sort([self.mR1sq,self.mSsq])
        #self.μs = np.random.choice([-1,1])*np.random.uniform(low=Lowmass**2,high=Highmass**2,)

        self.vh=246.
        self.initialVals = False
        self.cond = np.full(7,False,dtype=bool)

    def init(self,μr,μs,λr,λs,ghs,ghr,grs,ghr2,a1,mR1sq,mSsq, initialVals = True):
        """
        This fuctions aims to overwrite the values in __init__ if we want to change
        the limits
        """
        self.vh=246.
        #self.mR1sq, self.mR2sq, self.mSsq = mR1sq, mR1sq, mSsq
        self.ghs, self.ghr, self.grs,self.ghr2 = ghs, ghr, grs, ghr2
        self.μr,self.μs = μr, μs
        self.λr, self.λs =  λr, λs
        self.a1 = a1
        self.mR1sq, self.mSsq = mR1sq, mSsq
        self.cond = np.full(7,False,dtype=bool)
        self.initialVals = True

    #def gaugeVsLQMass(self):
    #    """
    #    This fuctioni is taken from Marco's code
    #    ========================================
    #    This fuction aims to calculate μs,μr,mSsq
    #    """
    #    ghs,ghr,ghr2,grs,a1,vh = self.ghs,self.ghr,self.ghr2,self.grs,self.a1,self.vh
    #    mR1sq,mR2sq=self.mR1sq, self.mR2sq
    #    self.μs = (2*a1**2*vh**2 + (2*mR1sq - 2*mR2sq + ghr2*vh**2)*(2*mR2sq - ghs*vh**2))/(4*mR1sq - 4*mR2sq + 2*ghr2*vh**2)
    #    self.μr = mR1sq - (ghr*vh**2)/2.
    #    self.mSsq = (2*a1**2*vh**2 + (2*mR1sq + ghr2*vh**2)*(2*mR1sq - 2*mR2sq + ghr2*vh**2))/(4*mR1sq - 4*mR2sq + 2*ghr2*vh**2)

    def gaugeVsLQMass(self):
        ghs,ghr,grs,a1,vh,ghr2 = self.ghs,self.ghr,self.grs,self.a1,self.vh,self.ghr2
        mR1sq, mSsq = self.mR1sq, self.mSsq
        #mSsq = self.mSsq

        #v = self.vh
        #λh = mHsq/(2*vh**2),
        #self.ghr2 = (-2*mCh + mRsq + mSsq + Sqrt((mRsq - mSsq)**2 - 2*a1**2*vh**2))/vh**2
        #self.μr = (mRsq + mSsq - (ghr + self.ghr2)*vh**2 + Sqrt((mRsq - mSsq)**2 - 2*a1**2*vh**2))/2
        #self.μs = (mRsq + mSsq - ghs*vh**2 - Sqrt((mRsq - mSsq)**2 - 2*a1**2*vh**2))/2
        #self.ghr2 = (-2*mCh*v**2 + mRsq*v**2 + mSsq*v**2 + Sqrt(-(v**4*(-mRsq**2 + 2*mRsq*mSsq - mSsq**2 + 2*v**2*a1**2))))/v**4
        #self.ghr = -self.ghr2 + (v**2*(mRsq + mSsq - 2*μr) + Sqrt((mRsq - mSsq)**2*v**4 - 2*v**6*a1**2))/v**4
        #self.ghs = (mRsq + mSsq - 2*μs - Sqrt((mRsq - mSsq)**2*v**4 - 2*v**6*a1**2)/v**2)/v**2
        self.μr = (mR1sq + mSsq - (ghr + ghr2)*vh**2 + np.sqrt((mR1sq - mSsq)**2 - 2*a1**2*vh**2))/2
        self.μs = (mR1sq + mSsq - ghs*vh**2 - np.sqrt((mR1sq - mSsq)**2 - 2*a1**2*vh**2))/2
        self.mR2sq = (mR1sq + mSsq - ghr2*vh**2 + np.sqrt((mR1sq - mSsq)**2 - 2*a1**2*vh**2))/2



    def InitiationValuesSM(self):
        """
        This fuction returns SM parameters (GeV) at µ = mZ = 91.1876 GeV, from arXiv 2009.04851v2
        From Marco's code
        """
        self.mZ = 91.1876
        self.gY = 0.357254 #+ np.random.uniform(low=-0.000069,high=0.000069)   #g1
        self.gw = 0.65100 #+ np.random.uniform(low=-0.00028,high=0.00028) #g2
        self.gs = 1.2104 #+ np.random.uniform(low=-0.0051,high=0.0051) #g3
        self.mt = 168.26 #+ np.random.uniform(low=-0.75,high=0.75)
        self.mb = 2.839 #+ np.random.uniform(low=-0.026,high=0.026)
        self.mτ = 1.72856 #+ np.random.uniform(low=-0.00028,high=0.00028)
        self.Yt = np.sqrt(2)*self.mt/self.vh
        self.Yb = np.sqrt(2)*self.mb/self.vh
        self.Yτ = np.sqrt(2)*self.mτ/self.vh
        self.λh = 0.13947 #+ np.random.uniform(low=-0.00045,high=0.00045)
        self.μh = -(self.vh**2*self.λh)
        self.mh = self.λh*(2*self.vh**2)

    def BFB1(self):
        """
        The conditions are taken from Marco's code

        """
        self.cond[0] = (self.λh>=0)
        self.cond[1] = (self.λs>=0)
        self.cond[2] = (self.λr>=0)

    def BFB2(self):
        self.cond[3] = (2*self.ghs + np.sqrt(self.λh*self.λs)>=0)
        self.cond[4] = (2*self.grs + np.sqrt(self.λr*self.λs)>=0)
        self.cond[5] = (2*self.ghr + 2*self.ghr2 + np.sqrt(self.λh*self.λr)>=0)
        self.cond[6] = (2*self.grs*np.sqrt(2)*np.sqrt(self.λh) + 2*self.ghs*np.sqrt(2)*np.sqrt(self.λr) +\
            2*self.ghr*np.sqrt(2)*np.sqrt(self.λs) + 2*self.ghr2*np.sqrt(2)*np.sqrt(self.λs) + \
            np.sqrt(2)*np.sqrt(self.λh*self.λr*self.λs) + 2*np.sqrt((2*self.ghr + 2*self.ghr2 +\
            np.sqrt(self.λh*self.λr))*(2*self.ghs + np.sqrt(self.λh*self.λs))*(2*self.grs +\
            np.sqrt(self.λr*self.λs)))>=0)

    def BFB3(self):
        self.Unitcond = np.full(7,False,dtype=bool)
        self.Unitcond = [abs(self.λh)<5*np.pi,abs(self.λs)<5*np.pi,abs(self.λr)<5*np.pi,
                        abs(self.ghs)<5*np.pi,abs(self.grs)<5*np.pi,abs(self.ghr)<5*np.pi,
                        abs(self.ghr2)<5*np.pi]
    def BFBm(self):
        self.mass = np.full(3,False,dtype=bool)
        self.mass[0] = (self.mSsq>= 0)
        self.mass[1] = (self.mR1sq>= 0)
        self.mass[2] = (self.mR2sq>= 0)

    def printValues(self):
        print("""===========================================================================""")
        print("""Leptoquark masses: mR1,mR2,mS = [{0},{1},{2}]""".format(self.mR1sq**(0.5), self.mR2sq**(0.5),self.mSsq**(0.5)))
        print("""Quartic couplings: λr, λs, ghs, ghr, ghr2, grs = [{0},{1},{2},{3},{4},{5}]""".format(self.λr, \
            self.λs,self.ghs, self.ghr, self.ghr2, self.grs))
        print("""mass terms: μs,μr = [{0},{1}]""".format(self.μs,self.μr))
        print("""Cubic coupling: a1 = [{0}]""".format(self.a1))
        print("""SM mass terms: μh = [{0}]
SM Quartic coupling: λh = [{1}]
SM yukawa couplings: Yt,Yb,Yτ = [{2},{3},{4}]
SM higgs mass: mh = [{5}]
Gauge couplingd: gs,gw,gY = [{6},{7},{8}]""".format(self.μh,self.λh,self.Yt,self.Yb,self.Yτ,self.mh,self.gs,self.gw,self.gY))

    def returnFunction(self):
        """
        This function returns all the need it values in order to be used in the
        next step
        """
        QuartNam = np.array(["λr", "λs", "ghs", "ghr", "ghr2", "grs","λh"])
        QuartVal = np.array([self.λr,self.λs,self.ghs, self.ghr, self.ghr2, self.grs,self.λh])
        CubicNam = np.array(["a1"])
        CubivVal = self.a1
        masstNam = np.array(["μs","μr","μh"])
        masstVal = np.array([self.μs,self.μr,self.μh])
        YukawNam = np.array(["Yt","Yb","Yτ"])
        YukawVal = np.array([self.Yt,self.Yb,self.Yτ])
        GaugCoupNam = np.array(["gs","gw","gY"])
        GaugCoupVal = np.array([self.gs,self.gw,self.gY])

        return [QuartNam,QuartVal,masstNam,masstVal,YukawNam,YukawVal,\
                GaugCoupNam,GaugCoupVal,CubicNam,CubivVal]


    def main(self):
        self.UnitPass = False
        if not self.initialVals:
            while not self.UnitPass:
                self.__init__(self.Lowmass, self.Highmass, self.Lowcoup, self.Highcoup, self.Lowcubic, \
                self.Highcubic)
                self.gaugeVsLQMass()
                self.InitiationValuesSM()
                self.BFBm()
                if sum(self.mass)==3:
                    self.BFB1()
                    if sum(self.cond[0:3]) == 3:
                        self.BFB2()
                        self.BFB3()
                        if sum(self.cond) == 7 and sum(self.Unitcond) == 7: self.UnitPass = True
            #self.printValues()
            #print("UnitPass:",UnitPass)
        if self.initialVals:
            self.gaugeVsLQMass()
            self.InitiationValuesSM()
            self.BFBm()
            if sum(self.mass)==3:
                self.BFB1()
                if sum(self.cond[0:3]) == 3:
                    self.BFB2()
                    self.BFB3()
                    if sum(self.cond) == 7 and sum(self.Unitcond) == 7: self.UnitPass = True

        #self.printValues()


class solBetafunc():
    """
    This class aims to solve the beta fuctions, taking as an input the tree level
    couplings.
    """
    def __init__(self):
        coup = gaugeCouplingsTreeLeve()
        coup.main()
        QuartNam,QuartVal,masstNam,masstVal,YukawNam,YukawVal,GaugCoupNam,GaugCoupVal,\
        CubicNam,CubivVal = coup.returnFunction()

        self.λr,self.λs,self.ghs, self.ghr, self.ghr2, self.grs,self.λh = QuartVal
        self.μs,self.μr,self.μh = masstVal
        self.Yt,self.Yb,self.Yτ = YukawVal
        self.μSM = 91.1876
        self.gs,self.gw,self.gY = GaugCoupVal
        self.a1 = CubivVal
        self.μRef = 1.0e3
        self.μMax = 1.0e4*Pi

    def init(self,QuartVal,masstVal,YukawVal,GaugCoupVal,CubivVal,μRef,μMax):
        self.λr,self.λs,self.ghs, self.ghr, self.ghr2, self.grs,self.λh = QuartVal
        self.μs,self.μr,self.μh = masstVal
        self.Yt,self.Yb,self.Yτ = YukawVal
        self.μSM = 91.1876
        self.gs,self.gw,self.gY = GaugCoupVal
        self.a1 = CubivVal
        self.μRef = μRef
        self.μMax = μMax #1.0e4*Pi # need to give that as input

    def SMbetaFunc(self,μ,params):
    	"""
        RG eqs for the SM 4D parameters: RHS of the equation dg/dμ = β(g)/μ.
    	params: {μh, λh, gYsq, gwsq, gssq, Yt, Yb, Yτ}
    	Returns:{μh, λh, gYsq, gwsq, gssq, Yt, Yb, Yτ} β functions.
        ==================================================
        From Marco's Code
        """
    	μh, λh, gYsq, gwsq, gssq, Yt, Yb, Yτ = params
    	βμh = ((4*Yτ**2 - 3*(3*gwsq + gYsq - 4*(Yb**2 + Yt**2 + 2*λh)))*μh)/(32.*Pi**2)
    	βλh = (-16*Yτ**4 + 32*Yτ**2*λh + 3*(3*gwsq**2 + gYsq**2 - 16*(Yb**2 + Yt**2)**2 + 2*gwsq*(gYsq - 12*λh) - 8*gYsq*λh + 32*λh*(Yb**2 + Yt**2 + 2*λh)))/(128.*Pi**2)
    	βgYsq = (41*gYsq**2)/(48.*Pi**2)
    	βgwsq = (-19*gwsq**2)/(48.*Pi**2)
    	βgssq = (-7*gssq**2)/(8.*Pi**2)
    	βYt = (Yt*(-96*gssq - 27*gwsq - 17*gYsq + 54*(Yb**2 + Yt**2) + 12*Yτ**2))/(192.*Pi**2)
    	βYb = (Yb*(-96*gssq - 27*gwsq - 5*gYsq + 54*(Yb**2 + Yt**2) + 12*Yτ**2))/(192.*Pi**2)
    	βYτ = (Yτ*(-9*gwsq - 15*gYsq + 12*(Yb**2 + Yt**2) + 10*Yτ**2))/(64.*Pi**2)
    	return [βμh/μ, βλh/μ, βgYsq/μ, βgwsq/μ, βgssq/μ, βYt/μ, βYb/μ, βYτ/μ]

    def DRBetaFunctions4D(self,μ, params4D):
        """
        Returns the RHS of the RG-equation dp/dμ = β(p)/μ.

        This function returns an array with the β-functions for the 4D-para-
        meters divided by the RG-scale μ, i.e. the right hand side in the RG-
        equation dp/dμ = β(p)/μ, where p denotes the array of 4D-parameters.

        Parameters
        ----------
        mu : float
            The 4D RG-scale parameter (i.e. μ)
        params4D : array
            Array of the 4D-parameters at scale μ

        Returns
        ----------
        RHS of th
        """

        gsSQ, gwSQ, gYSQ, ghr2, ghr, ghs, grs, λh, λr, λs, a1, Yb, Yt, Yτ, μh, μr, μs = params4D

        βgsSQ = (-13*gsSQ**2)/(16.*Pi**2) #*(1/(2*gs))
        βgwSQ = -0.3333333333333333*gwSQ**2/Pi**2  #*(1/(2*gw))
        βgYSQ = (7*gYSQ**2)/(8.*Pi**2) #*(1/(2*gY))
        βghr2 = (-((10*ghr2 + 3*Sqrt(3)*gwSQ)*gYSQ) + 6*ghr2*(8*ghr - 8*gsSQ - 9*gwSQ + 6*Yb**2 + 6*Yt**2 + 2*Yτ**2 + 4*(λh + λr)))/(96.*Pi**2)
        βghr = (48*ghr**2 + 48*ghr2**2 + 72*ghs*grs + 27*gwSQ**2 + gYSQ**2 + 4*ghr*(-24*gsSQ - 27*gwSQ - 5*gYSQ + 6*(3*Yb**2 + 3*Yt**2 + Yτ**2 + 6*λh + 14*λr)))/(192.*Pi**2)
        βghs = (24*ghs**2 + 72*ghr*grs + 2*gYSQ**2 + ghs*(-48*gsSQ - 27*gwSQ - 13*gYSQ + 12*(3*Yb**2 + 3*Yt**2 + Yτ**2 + 6*λh + 8*λs)))/(96.*Pi**2)
        βgrs = (216*ghr*ghs + 216*grs**2 + 99*gsSQ**2 + 12*gsSQ*gYSQ + 2*gYSQ**2 - 9*grs*(96*gsSQ + 27*gwSQ + 5*gYSQ - 168*λr - 96*λs))/(864.*Pi**2)
        βλh = (48*ghr**2 + 16*ghr2**2 + 24*ghs**2 + 9*gwSQ**2 + 3*gYSQ**2 - 48*(Yb**2 + Yt**2)**2 - 16*Yτ**4 + 6*gwSQ*(gYSQ - 12*λh) - 24*gYSQ*λh + 32*λh*(3*Yb**2 + 3*Yt**2 + Yτ**2 + 6*λh))/(128.*Pi**2)
        βλr = (432*ghr**2 - 144*ghr2**2 + 648*grs**2 + 198*gsSQ**2 + 243*gwSQ**2 + gYSQ**2 + 12*gsSQ*(63*gwSQ - gYSQ - 288*λr) - 72*gYSQ*λr + 8640*λr**2 - 18*gwSQ*(gYSQ + 108*λr))/(3456.*Pi**2)
        βλs = (108*ghs**2 + 324*grs**2 + 117*gsSQ**2 + 24*gsSQ*(gYSQ - 36*λs) + 4*(gYSQ**2 - 18*gYSQ*λs + 378*λs**2))/(864.*Pi**2)
        βa1 = (a1*(12*ghr - 12*Sqrt(3)*ghr2 + 12*ghs + 12*grs - 48*gsSQ - 27*gwSQ - 7*gYSQ + 6*(3*(Yb**2 + Yt**2) + Yτ**2)))/(96.*Pi**2)
        βYb = (Yb*(-96*gsSQ - 27*gwSQ - 5*gYSQ + 54*(Yb**2 + Yt**2) + 12*Yτ**2))/(192.*Pi**2)
        βYt = (Yt*(-96*gsSQ - 27*gwSQ - 17*gYSQ + 54*(Yb**2 + Yt**2) + 12*Yτ**2))/(192.*Pi**2)
        βYτ = (Yτ*(-9*gwSQ - 15*gYSQ + 12*(Yb**2 + Yt**2) + 10*Yτ**2))/(64.*Pi**2)
        βμh = (12*a1**2 - 9*gwSQ*μh - 3*gYSQ*μh + 4*((3*Yb**2 + 3*Yt**2 + Yτ**2 + 6*λh)*μh + 6*ghr*μr + 3*ghs*μs))/(32.*Pi**2)
        βμr = (12*a1**2 + 24*ghr*μh - (48*gsSQ + 27*gwSQ + gYSQ - 168*λr)*μr + 36*grs*μs)/(96.*Pi**2)
        βμs = (6*(a1**2 + ghs*μh + 3*grs*μr) - (12*gsSQ + gYSQ - 24*λs)*μs)/(24.*Pi**2)

        return [βgsSQ/μ, βgwSQ/μ, βgYSQ/μ, βghr2/μ, βghr/μ, βghs/μ, βgrs/μ, βλh/μ, βλr/μ, βλs/μ, βa1/μ, βYb/μ, βYt/μ, βYτ/μ, βμh/μ, βμr/μ, βμs/μ]


    def main(self):
        """
        In the main function solve the SM beta fuctions from mZ scale up to the
        reference scale. Then we also solve the LQ beta function up to the μMax
        """
        SMparm = np.array([self.μh, self.λh, self.gY**2, self.gw**2, self.gs**2, \
                self.Yt, self.Yb, self.Yτ])
        """
        leftSol : Soving beta functions for SM scale (mZ) to Leptoquark model scale
        """
        leftSol = solve_ivp(self.SMbetaFunc, [self.μSM,self.μRef],
                                    SMparm, dense_output = True)
        self.smsol = OdeSolution(leftSol.t,leftSol.sol.interpolants)
        """
        Prepering the LQ couplings
        """
        μh, λh, gYsq, gwsq, gssq, Yt, Yb, Yτ= self.smsol(self.μRef)

        self.params4D = np.array([gssq, gwsq, gYsq, self.ghr2, self.ghr, self.ghs, self.grs,\
                    λh, self.λr, self.λs, self.a1, Yb, Yt, Yτ, μh, self.μr, self.μs])

        """
        rightSol : Soving beta functions for LQ
        """
        rightSol = solve_ivp(self.DRBetaFunctions4D, [self.μRef, self.μMax],
                                self.params4D, dense_output = True)
        self.lqsol = OdeSolution(rightSol.t,rightSol.sol.interpolants)
        self.lqsol = np.vectorize(self.lqsol, signature='()->(n)')


class DRalgoModel(generic_potential.generic_potential):
    def __init__(self, *args, **dargs):
        self.Ndim = 0
        self.x_eps = .001
        self.T_eps = .001
        self.deriv_order = 4
        self.renormScaleSq = 1000.**2
        self.Tmax = 1e3

        self.num_boson_dof = self.num_fermion_dof = None

        self.phases = self.transitions = None  # These get set by getPhases
        self.TcTrans = None  # Set by calcTcTrans()
        self.TnTrans = None  # Set by calcFullTrans()
        self.VEff3D = DR.Veff3D

        self.init(*args, **dargs)

        if self.Ndim <= 0:
            raise ValueError("The number of dimensions in the potential must "
                             "be at least 1.")
    def init(self,lqsol=None,muRef=None,muMax=None,scaleFactor=None,NLO=None):
        """
        bla bla
        """
        if NLO is None: self.NLO=False
        else: self.NLO = NLO
        if muRef is None: self.muRef = 1.0e3 # 1 Tev
        else:self.muRef = muRef
        if muMax is None: self.muMax = 1.0e4 * np.pi
        else: self.muMax = muMax
        if scaleFactor is None: self.scaleFactor = 1.0
        else: self.scaleFactor = scaleFactor
        if lqsol is None: self.lqsol = DR.InitiationValuesSM()
        else: self.lqsol = lqsol
        """
        Fix tempreturea and internal scales
        """
        self.Tmax = self.muMax/(self.scaleFactor* np.pi)
        self.Tmin = self.muRef/(self.scaleFactor* np.pi)
        self.Ndim = 3

    def getPhases(self,tracingArgs={}):
        """
        Find different phases as functions of temperature

        Parameters
        ----------
        tracingArgs : dict
            Parameters to pass to :func:`transitionFinder.traceMultiMin`.

        Returns
        -------
        dict
            Each item in the returned dictionary is an instance of
            :class:`transitionFinder.Phase`, and each phase is
            identified by a unique key. This value is also stored in
            `self.phases`.
        """
        tstop = self.Tmax
        points = []
        for x0 in self.approxZeroTMin():
            points.append([x0,self.Tmin])
        tracingArgs_ = dict(forbidCrit=self.forbidPhaseCrit)
        tracingArgs_.update(tracingArgs)
        phases = transitionFinder.traceMultiMin(
            self.Vtot, self.dgradV_dT, self.d2V, points,
            tLow=self.Tmin, tHigh=tstop, deltaX_target=100*self.x_eps,
            **tracingArgs_)
        self.phases = phases
        transitionFinder.removeRedundantPhases(
            self.Vtot, phases, self.x_eps*1e-2, self.x_eps*10)
        return self.phases

    def forbidPhaseCrit(self, X):
        """
        forbidPhaseCrit if a X is larger than 10^6 it doesn't stops it.
        """
        return (abs(np.array([X])) > 10.0e15).any()


    def Vtot(self,X,T,include_radiation=True):
        """
        The finite temperature effective potential.

        This function overrides the function Vtot in the generic_potential
        class of CosmoTransitions. It returns the *real part only* of the
        effective potential for the given field values and at the given
        temperature. More specifically, it returns the real part of T*VEff3D,
        where VEff3D is the 3D effective potential at the given temperature
        and with the given vevs. Thus, the return value has mass dimension 4.


        Parameters
        ----------
        X : array_like
            Field value(s).
            Either a single point (with length `Ndim`), or an array of points.
        T : float or array_like
            The temperature. The shapes of `X` and `T`
            should be such that ``X.shape[:-1]`` and ``T.shape`` are
            broadcastable (that is, ``X[...,0]*T`` is a valid operation).
        include_radiation : bool, optional
            NOTE: This is kept for compatibility reasons with CosmoTransitions.
            Setting this parameter to True or False does not change anything.

        Returns
        -------
        The (real part of the) effective potential
        """
        #print("X:",X,"T:",T)
        μ = self.scaleFactor *Pi * T
        gsSQ, gwSQ, gYSQ, ghr2, ghr, ghs, grs, λh, λr, λs, a1, Yb, Yt, Yτ, μh, \
        μr, μs = self.lqsol(μ).reshape(17,)

        sol4dBeta = np.array([gsSQ**(1/2),gwSQ**(1/2),gYSQ**(1/2),ghr2, ghr, ghs, grs, λh, λr, λs, a1, Yb, Yt, Yτ, μh, μr, μs],dtype='float64')
        μ3 = DR.DRPrintDebyeMass(sol4dBeta,μ,T,self.NLO)[1]
        μ3US = μ3
        vh, vs, vr = X[...,0], X[...,1], X[...,2]
        #print("params4DVsMu(μ)",self.params4DVsMu(μ))
        # Veff3D(sol4dBeta,μ,μ3,μ3US,X,T,NLO):
        #print("X.shape",X.shape)
        #print("T",T)
        #print(type(T))
        #print("μ.shape",μ.shape)
        #print("sol4dBeta.shape",sol4dBeta.shape)
        veff = self.VEff3D(sol4dBeta,μ,μ3,μ3US,X,T,self.NLO)

        #params4D = self.params4DVsMu(mu).reshape(len(self.params4DRef),)

        return veff.real.astype('float64')


if __name__ == "__main__":
    print("Starting")
    phase = 0
    pass1 = False
    while phase <50:
    #for phase in range(0,1):
        CTbrakes = False
        coup = gaugeCouplingsTreeLeve()
        initialVals = False
        Lowmass=0.2e03
        Highmass=4.0e03
        Lowcoup=-4*np.pi
        Highcoup=4*np.pi
        Lowcubic=1.0e2
        Highcubic=1.0e3
        if initialVals:
            ghs, ghr, grs, ghr2  = np.random.uniform(low=Lowcoup,high=Highcoup,size=4)
            λr, λs =  np.random.uniform(low=0,high=Highcoup,size=2)

            a1 = np.random.choice([-1,1])*np.random.uniform(low=Lowcubic,high=Highcubic)
            μr = np.random.choice([-1,1])*np.random.uniform(low=Lowmass,high=Highmass)
            μs = np.random.choice([-1,1])*np.random.uniform(low=Lowmass,high=Highmass)
            coup.init(μr,μs,λr,λs,ghs,ghr,grs,ghr2,a1, initialVals = initialVals)
        if not initialVals:
            coup.__init__(Lowmass=Lowmass, Highmass=Highmass, Lowcoup=Lowcoup, Highcoup=Highcoup,\
                Lowcubic=Lowcubic, Highcubic=Highcubic)
        coup.main()
        if coup.UnitPass:
            coup.printValues()
            print("UnitPass : {}".format(coup.UnitPass))
            QuartNam,QuartVal,masstNam,masstVal,YukawNam,YukawVal,GaugCoupNam,GaugCoupVal,\
            CubicNam,CubivVal = coup.returnFunction()
            solb = solBetafunc()
            μRef= np.sqrt(np.sum([coup.mR1sq,coup.mR2sq,coup.mSsq]))/3.0
            μMax=2.0e4*Pi
            solb.init(QuartVal,masstVal,YukawVal,GaugCoupVal,CubivVal,μRef,μMax)

            solb.main()
            print("μRef:",μRef)
            ct = DRalgoModel()
            ct.init(lqsol = solb.lqsol,muRef=μRef,muMax=μMax,scaleFactor=1.0,NLO=False)
            print("""=================================""")
            print("""Starting CT""")
            print("""=================================""")
            CTbrakes = False
            try:
                ct.findAllTransitions()
                print("TnTrans :",ct.TnTrans)
            except:
                CTbrakes = True
            if not CTbrakes:
                ct.prettyPrintTnTrans()
                if len(ct.TnTrans)>0:
                    phase +=1 #print("=============phase")
                    print("TnTrans :",ct.TnTrans)
                    savingOp = True
                    for i in range(len(ct.TnTrans)):
                        if ct.TnTrans[i]['trantype']==1:
                            savingOp = False
                            dill.dump(ct, file = open(os.path.join(outputFol,"1stCTLQ_{}.pt".format(phase)),"wb"))
                            dill.dump(solb, file = open(os.path.join(outputFol,"1stsolbLQ_{}.pt".format(phase)),"wb"))
                            dill.dump(coup, file = open(os.path.join(outputFol,"1stcoupLQ_{}.pt".format(phase)),"wb"))
                    if savingOp:
                        dill.dump(ct, file = open(os.path.join(outputFol,"CTLQ_{}.pt".format(phase)),"wb"))
                        dill.dump(solb, file = open(os.path.join(outputFol,"solbLQ_{}.pt".format(phase)),"wb"))
                        dill.dump(coup, file = open(os.path.join(outputFol,"coupLQ_{}.pt".format(phase)),"wb"))
