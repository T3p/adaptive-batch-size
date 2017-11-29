from exp_lqg1d import run

#First experiment
trial = '1'
run(0.95,'chebyshev','reinforce',filename='chebyshev_reinforce_095_'+trial +'.h5')
run(0.75,'chebyshev','reinforce',filename='chebyshev_reinforce_075_'+trial +'.h5')
run(0.5,'chebyshev','reinforce',filename='chebyshev_reinforce_050_'+trial +'.h5')
run(0.95,'chebyshev','gpomdp',filename='chebyshev_gpomdp_095_'+trial +'.h5')
run(0.75,'chebyshev','gpomdp',filename='chebyshev_gpomdp_075_'+trial +'.h5')
run(0.5,'chebyshev','gpomdp',filename='chebyshev_gpomdp_050_'+trial +'.h5')
run(0.25,'chebyshev','gpomdp',filename='chebyshev_gpomdp_025_'+trial +'.h5')
run(0.05,'chebyshev','gpomdp',filename='chebyshev_gpomdp_005_'+trial +'.h5')

#Second experiment
trial = '1'
run(0.95,'hoeffding','gpomdp',emp=False,filename='hoeffding_theo_'+trial+'.h5')
run(0.95,'hoeffding','gpomdp',emp=True,filename='hoeffding_emp_'+trial+'.h5')
run(0.95,'bernstein','gpomdp',emp=False,filename='bernstein_theo_'+trial+'.h5')
run(0.95,'bernstein','gpomdp',emp=True,filename='hoeffding_emp_'+trial+'.h5')











