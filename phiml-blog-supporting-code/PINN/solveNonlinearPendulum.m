function sol = solveNonlinearPendulum(tspan,omega0,thetaDot0)
    F = ode;
    F.ODEFcn = @(t,x) [x(2); -omega0^2.*sin(x(1))];
    F.InitialValue = [0; thetaDot0];
    F.Solver = "ode45";
    sol = solve(F,tspan);
end