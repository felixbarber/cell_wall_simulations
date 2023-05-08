import numpy as np
import scipy
from scipy import stats
from scipy.optimize import fsolve
import math


def converges(temp_x, window=3):
    # this function takes as input a vector,
    # and reports both whether or not that vector appears to be converging,
    # and what the final gradient of that vector is
    # We will approach this by calculating the local gradient of elements in that sequence,
    # and whether the trend of this sequence of gradients is decreasing or not.

    # This function returns a binary statistic of convergence or not, the final slope of the sequence, and the sign
    # of the direction it moves
    temp_out = np.empty(temp_x.shape)
    for i0 in range(len(temp_x)):
        temp_vec = temp_x[np.amax([0, i0 - window]):np.amin([len(temp_x) - 1, i0 + window])]
        temp_grad = scipy.stats.linregress(np.arange(len(temp_vec)), temp_vec)[0]
        #         print(temp_grad)
        temp_out[i0] = np.absolute(temp_grad)
    # Having calculated the absolute local gradient of the sequence, we will see whether this is converging to zero.
    # We do this by calculating the supremum at each point and seeing whether this value tends towards zero in
    # a reasonable time.
    temp_out1 = np.empty(temp_x.shape)
    for i0 in range(len(temp_x)):
        temp_out1[i0] = np.amax(temp_out[i0:])
    # At first I tried a linear regression test for convergence, but this only seemed to work for monotonic sequences.
    #     temp_vals=scipy.stats.linregress(np.arange(len(temp_out1)),temp_out1)
    #     xint=-temp_vals[1]/temp_vals[0]
    #     if xint>0 and xint<1000:
    #         conv=1
    #     else:
    #         conv=0
    if temp_out[-1] < 0.001 * temp_x[-1]:
        conv = 1
    else:
        conv = 0
    return conv, temp_out[-1], np.absolute(conv - 1) * (np.sign(temp_x[-1] - temp_x[0]))


def insertion_rate(r, r_opt=0.5, feedback=10, opt_rate=1.0):
    # This function gives the insertion rate for new layers of cell wall in layers per minute

    #     temp1=opt_rate/(1+feedback*(np.absolute(r-r_opt))**2)
    temp1 = opt_rate
    return temp1


def eq_sys_grad_hydr(temp_vals, temp_ri, temp_li, temp_si, temp_params):
    # this function takes a vector of rest lengths, rest radii and stiffnesses, and
    # calculates the strain in each case based on inputs for the overall cell inner radius and length,
    # temp_rval and temp_lval.
    # Here temp_vals is the proposed internal radius and cell length, while temp_ri and temp_li
    # are the rest radius and rest length for each layer.
    # Note that this was intended for the gradual hydrolysis case, which I didn't pursue very far.
    temp_dr = temp_params['dr']
    temp_alpha = temp_params['alpha']
    temp_nu = temp_params['nu']
    temp_PK_rat = temp_params['PK_rat']

    temp_rval = temp_vals[0]
    temp_lval = temp_vals[1]
    # We now calculate the radial strain = (r1-r0)/r0 and longitudinal strain = (l1-l0)/l0
    eps_r = (temp_rval + temp_dr * np.arange(0, len(temp_ri)).astype(
        float) - temp_ri) / temp_ri  # radial strain in each layer
    # based on equal spacing between layers of dr.
    eps_l = (temp_lval - temp_li) / temp_li  # longitudinal strain in each layer

    # Now we calculate the tensions in each layer
    sig_l = temp_dr * temp_si * ((temp_alpha / (temp_alpha - temp_nu ** 2)) * eps_l + (
                temp_nu / (temp_alpha - temp_nu ** 2)) * eps_r)  # the longitudinal tension in each layer
    sig_r = temp_dr * temp_si * (
                (temp_nu / (temp_alpha - temp_nu ** 2)) * eps_l + (1 / (temp_alpha - temp_nu ** 2)) * eps_r)
    # the radial tension in each layer
    # note the tension is normalized by the spacing of each layer. We assume each layer to have thickness dr.

    temp_out1 = np.sum(sig_r) - temp_PK_rat * np.absolute(
        temp_rval)  # this is the first constraint equation (this should equal 0)
    temp_out2 = np.sum((np.absolute(temp_rval) + temp_dr * np.arange(0, len(temp_ri)).astype(float)) * sig_l) - \
                temp_PK_rat * (np.absolute(temp_rval) ** 2) / 2
    # This is the second constraint equation which should also equal zero.
    return [temp_out1, temp_out2]


def sim_growth(temp_tvec, temp_ri, temp_li, temp_si, temp_params):
    # This function performs gradual hydrolysis, but has no radial dependency of insertion rate or initial cell wall
    # stiffness on curvature of the cell wall. This is changed in subsequent versions.
    temp_radii, temp_lengths, temp_stiffnesses = [temp_ri.copy()], [temp_li.copy()], [temp_si.copy()]
    insertion_times = []  # vector of timepoints at which new layers are inserted
    temp_eff_rad, temp_eff_length = np.empty(temp_tvec.shape), np.empty(temp_tvec.shape)
    # vector of equilibrium radii at each timepoint
    numerical_error = 0.000000001

    temp_hydr_rate = temp_params['hydr_rate']
    temp_dt = temp_params['dt']
    temp_dr = temp_params['dr']
    temp_synth_rate = temp_params['synth_rate']

    for ind in np.arange(0, len(temp_tvec)):
        # Let's now remove a layer, add an inside layer at the previous equilibrium radius,
        # and then equilibriate again.
        rint = temp_radii[-1].copy()  # This is the basic state, where effectively nothing happens.
        lint = temp_lengths[-1].copy()
        sint = temp_stiffnesses[-1].copy()
        insertion = True  # dummy variable for now
        if ind == 0:
            # for the very first timestep, all we do is equilibrate and figure out when we are next inserting
            # a layer
            rprev = temp_radii[-1][0].copy()
            lprev = temp_lengths[-1][0].copy()
            root = fsolve(eq_sys_grad_hydr, [rprev, lprev], args=tuple([rint, lint, sint, temp_params]))

        else:
            sint = sint - temp_dt * temp_hydr_rate
            if np.absolute(sint[-1]) < numerical_error:  # if one layer now has effectively zero stiffness
                sint = sint[:-1]  # in this case we remove the outer layer. Note that only one layer should ever hit
                # zero stiffness at one time.
                rint = rint[:-1]
                lint = lint[:-1]
            insertion = -0.5 < (temp_tvec[ind] - insertion_times[
                -1]) / temp_dt <= 0.5  # if an insertion should have taken place within this window
            if insertion:
                # insert a new layer at a radius equal to the previous equilibrium radius, at the previous eq.
                # length, with full stiffness.
                rint = np.concatenate((np.array([root[0] - temp_dr]), rint.copy()))
                lint = np.concatenate((np.array([root[1]]), lint.copy()))
                sint = np.concatenate((np.array([1.0]), sint.copy()))

            #         print(sint)
            # Now we update the equilibrium radius and length
            rprev = temp_radii[-1][0].copy()
            lprev = temp_lengths[-1][0].copy()
            root = fsolve(eq_sys_grad_hydr, [rprev, lprev], args=tuple([rint, lint, sint, temp_params]))

        if insertion:
            insertion_times.append(temp_tvec[ind] + 1.0 / insertion_rate(root[0], opt_rate=temp_synth_rate))
        temp_eff_rad[ind], temp_eff_length[ind] = root[0], root[1]
        # Updating the stored radii, lengths and stiffnesses
        temp_radii.append(rint.copy())
        temp_lengths.append(lint.copy())
        temp_stiffnesses.append(sint.copy())
    return temp_radii, temp_lengths, temp_stiffnesses, temp_eff_rad, temp_eff_length


def eq_sys_grad_hydr_v1(temp_vals, temp_ri, temp_li, temp_si, temp_params):
    # this function takes a vector of rest lengths, rest radii and stiffnesses, and
    # calculates the strain in each case based on inputs for the overall cell inner radius and length,
    # temp_rval and temp_lval. This function is updated to account for possible variability in the initial stiffness
    # of each layer.
    # Here temp_vals is the proposed internal radius and cell length, while temp_ri and temp_li
    # are the rest radius and rest length for each layer.
    # Note that this was intended for the gradual hydrolysis case, which I didn't pursue very far.
    temp_dr = temp_params['dr']
    temp_alpha = temp_params['alpha']
    temp_nu = temp_params['nu']
    temp_PK_rat = temp_params['PK_rat']

    temp_rval = temp_vals[0]
    temp_lval = temp_vals[1]
    # We now calculate the radial strain = (r1-r0)/r0 and longitudinal strain = (l1-l0)/l0
    eps_r = (temp_rval + temp_dr * np.arange(0, len(temp_ri)).astype(
        float) - temp_ri) / temp_ri  # radial strain in each layer
    # based on equal spacing between layers of dr.
    eps_l = (temp_lval - temp_li) / temp_li  # longitudinal strain in each layer

    # Now we calculate the tensions in each layer
    sig_l = temp_dr * temp_si * ((temp_alpha / (temp_alpha - temp_nu ** 2)) * eps_l + (
                temp_nu / (temp_alpha - temp_nu ** 2)) * eps_r)  # the longitudinal tension in each layer
    sig_r = temp_dr * temp_si * (
                (temp_nu / (temp_alpha - temp_nu ** 2)) * eps_l + (1 / (temp_alpha - temp_nu ** 2)) * eps_r)
    # the radial tension in each layer
    # note the tension is normalized by the spacing of each layer. We assume each layer to have thickness dr.

    temp_out1 = np.sum(sig_r) - temp_PK_rat * np.absolute(
        temp_rval)  # this is the first constraint equation (this should equal 0)
    temp_out2 = np.sum((np.absolute(temp_rval) + temp_dr * np.arange(0, len(temp_ri)).astype(float)) * sig_l) - \
                temp_PK_rat * (np.absolute(temp_rval) ** 2) / 2
    # This is the second constraint equation which should also equal zero.
    return [temp_out1, temp_out2]


def eq_sys1(temp_vals, temp_ri, temp_li, temp_params):
    # Note that this is the equation system to be solved for equilibrium at each step in the simulation
    # for the non-strain-stiffening, all-in-one hydrolysis model.

    # this function takes a vector of rest lengths and rest radii and calculates the strain in each case
    # based on inputs for the overall cell inner radius and length, temp_rval and temp_lval.
    # Here temp_vals is the proposed internal radius and cell length, while temp_ri and temp_li
    # are the rest radius and rest length for each layer. temp_params is a dictionary that contains various parameters
    # for the model.
    temp_rval = temp_vals[0]
    temp_lval = temp_vals[1]

    temp_dr = temp_params['dr']
    temp_alpha = temp_params['alpha']
    temp_nu = temp_params['nu']
    temp_PK_rat = temp_params['PK_rat']

    # We now calculate the radial strain = (r1-r0)/r0 and longitudinal strain = (l1-l0)/l0
    eps_r = (temp_rval + temp_dr * np.arange(0, len(temp_ri)).astype(
        float) - temp_ri) / temp_ri  # radial strain in each layer
    # based on equal spacing between layers of dr.
    eps_l = (temp_lval - temp_li) / temp_li  # longitudinal strain in each layer

    # Now we calculate the tensions in each layer
    sig_l = (temp_dr / (temp_alpha - temp_nu ** 2)) * (temp_alpha * eps_l + temp_nu * eps_r)
    # the longitudinal tension in each layer
    sig_r = (temp_dr / (temp_alpha - temp_nu ** 2)) * (temp_nu * eps_l + eps_r)
    # the radial tension in each layer
    # note the tension is normalized by the spacing of each layer. We assume each layer to have thickness dr.
    temp_out1 = np.sum(sig_r) - temp_PK_rat * np.absolute(temp_rval)  # note that we use absolute to avoid spurious
    # negative solutions, since radius should always be positive.
    # this is the first constraint equation (this should equal 0)
    temp_out2 = np.sum((np.absolute(temp_rval) + temp_dr * np.arange(0, len(temp_ri)).astype(float)) * sig_l) \
                - temp_PK_rat * (np.absolute(temp_rval) ** 2) / 2
    # This is the second constraint equation which should also equal zero.
    return [temp_out1, temp_out2]


def sim_growth_abrupt(temp_tvec, temp_ri, temp_li, temp_params):
    # This function takes the initial lengths and radii and simulates growth according to the abrupt, "all in one" model
    # of cell wall hydrolysis.
    # As parameters, this takes the time vector, the rest initial radii, the rest initial lengths and the additional
    # parameters
    temp_dr = temp_params['dr']
    # Spacing between each layer.
    temp_eff_rad, temp_eff_length = np.empty(temp_tvec.shape), np.empty(temp_tvec.shape)
    temp_eff_rad[0], temp_eff_length[0] = temp_ri[0],temp_li[0]
    # Rest lengths at each timepoint.
    # Now we perform the iteration
    temp_radii, temp_lengths = [temp_ri.copy()], [temp_li.copy()]
    for ind in range(len(temp_tvec)-1):
        if ind == 0:
            rint = temp_radii[-1].copy()  # initially we just equilibrate
            lint = temp_lengths[-1].copy()
        else:
            # Let's now remove a layer, add an inside layer at the previous equilibrium radius, and then equilibrate again.
            rint = np.concatenate(
                (np.array([root[0] - temp_dr]), temp_radii[-1][:-1].copy()))  # new layer inside the prior rest radius
            lint = np.concatenate(
                (np.array([root[1]]), temp_lengths[-1][:-1].copy()))  # new layer has the prior rest length
        rprev = rint[0].copy()
        lprev = lint[0].copy()
        root = fsolve(eq_sys1, [rprev, lprev], args=tuple([rint, lint, temp_params]))
        # This gives us the radius and length at which a new layer is added, and then we have a further round of
        # removal
        temp_radii.append(rint.copy())
        temp_lengths.append(lint.copy())
        temp_eff_rad[ind+1], temp_eff_length[ind+1] = root[0], root[1]
    return temp_radii, temp_lengths, temp_eff_rad, temp_eff_length


def sim_growth_abrupt_v2(temp_tvec, temp_ri, temp_li, temp_params):
    # This function takes the initial lengths and radii and simulates growth according to the abrupt, "all in one" model
    # of cell wall hydrolysis. However, it differs from sim_growth_abrupt_v1 in that it specifically makes sure that the
    # cell balances its wall stresses prior to inserting a new layer, which is what is needed to match up with theory.

    # As parameters, this takes the time vector, the rest initial radii, the rest initial lengths and the additional
    # parameters
    # time between each step of layer removal and insertion, in minutes. Usually set to 1.
    temp_dr = temp_params['dr']
    # Spacing between each layer.
    temp_eff_rad, temp_eff_length = np.empty(temp_tvec.shape), np.empty(temp_tvec.shape)
    temp_eff_rad[0], temp_eff_length[0] = temp_ri[0], temp_li[0]
    # Rest lengths at each timepoint.
    # Now we perform the iteration
    temp_radii, temp_lengths = [temp_ri.copy()], [temp_li.copy()]
    for ind in range(len(temp_tvec)-1):
        # if ind == 0:
        #     rint = temp_radii[-1].copy()  # initially we just equilibrate
        #     lint = temp_lengths[-1].copy()
        # else:
        # Let's now remove a layer
        rint = temp_radii[-1][:-1].copy()
        lint = temp_lengths[-1][:-1].copy()

        rprev = rint[0].copy()
        lprev = lint[0].copy()
        # Now we equilibriate to find the innermost radius
        root = fsolve(eq_sys1, [rprev, lprev], args=tuple([rint, lint, temp_params]))
        # Now we add a new layer inside the innermost radius
        rint = np.concatenate(
            (np.array([root[0] - temp_dr]), rint))  # new layer inside new rest radius
        lint = np.concatenate(
            (np.array([root[1]]), lint))  # new layer has the new rest length.

        # Note that as soon as we add this layer, the minimum energy configuration will change slightly. However, that
        # doesn't really matter since the rest lengths are still as defined, and we determine the next iteration of
        # layer addition based on the equilibrium state at the time.
        temp_radii.append(rint.copy())
        temp_lengths.append(lint.copy())
        temp_eff_rad[ind+1], temp_eff_length[ind+1] = root[0] - temp_dr, root[1]
    return temp_radii, temp_lengths, temp_eff_rad, temp_eff_length


def convergence_v1(temp_rad):
    # Crude test of convergence. 1 gives increasing radius, -1 indicates decreasing radius, nan indicates it's
    # indeterminate, 0 indicates stable
    rstep = np.diff(temp_rad[1:])
    signs = rstep > 0
    inds = np.nonzero(~signs)[0]
    if len(inds) > 0 and inds[0] > 1:  # this accounts for the case where it is initially increasing, then suddenly
        # decreases
        out = 1
    elif temp_rad[-1]-temp_rad[0] > 0.2 * temp_rad[0]:
        out = 1
    elif -temp_rad[-1]+temp_rad[0] > 0.2 * temp_rad[0]:
        out = -1
    else:
        out = 0
    return out



#######
# These functions simulate the strain-stiffening model

def eq_sys_strain_stiffening(temp_vals, temp_ri, temp_li, temp_params):
    # This equation system must be solved in each step of the simulations for the strain-stiffening model.
    # this function takes a vector of rest lengths and rest radii and calculates the strain in each case
    # based on inputs for the overall cell inner radius and length, temp_rval and temp_lval.
    # Here temp_vals is the proposed internal radius and cell length, while temp_ri and temp_li
    # are the rest radius and rest length for each layer. temp_params is a set of other simulation parameters.
    temp_rval = temp_vals[0]
    temp_lval = temp_vals[1]

    temp_dr = temp_params['dr']
    temp_alpha1 = temp_params['alpha1']
    temp_alpha2 = temp_params['alpha2']
    temp_strain_cutoff = temp_params['strain_cutoff']
    temp_nu = temp_params['nu']
    temp_PK_rat = temp_params['PK_rat']

    # We now calculate the radial strain = (r1-r0)/r0 and longitudinal strain = (l1-l0)/l0
    eps_r = (temp_rval + temp_dr * np.arange(0, len(temp_ri)).astype(
        float) - temp_ri) / temp_ri  # radial strain in each layer
    # based on equal spacing between layers of dr.
    eps_l = (temp_lval - temp_li) / temp_li  # longitudinal strain in each layer

    temp_alphas = temp_alpha1 * (eps_r <= temp_strain_cutoff) + temp_alpha2 * (
                eps_r > temp_strain_cutoff)  # this gives the
    # specific values for alpha for each layer in this configuration. Note that this is a vector.

    # Now we calculate the tensions in each layer
    sig_l = (temp_dr / (temp_alphas - temp_nu ** 2)) * (temp_alphas * eps_l + temp_nu * eps_r)
    # the longitudinal tension in each layer
    sig_r = (temp_dr / (temp_alphas - temp_nu ** 2)) * (temp_nu * eps_l + eps_r)
    # the radial tension in each layer
    # note the tension is normalized by the spacing of each layer. We assume each layer to have thickness dr.
    temp_out1 = np.sum(sig_r) - temp_PK_rat * np.absolute(temp_rval)  # note that we use absolute to avoid spurious
    # negative solutions, since radius should always be positive.
    # this is the first constraint equation (this should equal 0)
    temp_out2 = np.sum((np.absolute(temp_rval) + temp_dr * np.arange(0, len(temp_ri)).astype(float)) * sig_l) \
                - temp_PK_rat * (np.absolute(temp_rval) ** 2) / 2
    # This is the second constraint equation which should also equal zero.
    return [temp_out1, temp_out2]


def sim_growth_abrupt_strain_stiffening(temp_tvec, temp_ri, temp_li, temp_params):
    # This function takes the initial lengths and radii and simulates growth according to the abrupt, "all in one" model
    # of cell wall hydrolysis, subject to a "strain-stiffening" where the anisotropy changes from alpha1->alpha2 midway
    # through the cell wall based on the amount of radial strain.
    # This simulation specifically makes sure that the cell balances its wall stresses prior to
    # inserting a new layer
    # As parameters, this takes the time vector, the rest initial radii, the rest initial lengths and the additional
    # parameters
    # time between each step of layer removal and insertion, in minutes. Usually set to 1.
    temp_dr = temp_params['dr']
    # Spacing between each layer.
    temp_eff_rad, temp_eff_length = np.empty(temp_tvec.shape), np.empty(temp_tvec.shape)
    temp_eff_rad[0], temp_eff_length[0] = temp_ri[0], temp_li[0]
    # Rest lengths at each timepoint.
    # Now we perform the iteration
    temp_radii, temp_lengths = [temp_ri.copy()], [temp_li.copy()]
    for ind in range(len(temp_tvec)-1):
        # if ind == 0:
        #     rint = temp_radii[-1].copy()  # initially we just equilibrate
        #     lint = temp_lengths[-1].copy()
        # else:
        # Let's now remove a layer
        rint = temp_radii[-1][:-1].copy()
        lint = temp_lengths[-1][:-1].copy()

        rprev = rint[0].copy()
        lprev = lint[0].copy()
        # Now we equilibriate to find the innermost radius
        root = fsolve(eq_sys_strain_stiffening, [rprev, lprev], args=tuple([rint, lint, temp_params]))
        # Now we add a new layer inside the innermost radius
        rint = np.concatenate(
            (np.array([root[0] - temp_dr]), rint))  # new layer inside new rest radius
        lint = np.concatenate(
            (np.array([root[1]]), lint))  # new layer has the new rest length.

        # Note that as soon as we add this layer, the minimum energy configuration will change slightly. However, that
        # doesn't really matter since the rest lengths are still as defined, and we determine the next iteration of
        # layer addition based on the equilibrium state at the time.
        temp_radii.append(rint.copy())
        temp_lengths.append(lint.copy())
        temp_eff_rad[ind+1], temp_eff_length[ind+1] = root[0] - temp_dr, root[1]
    return temp_radii, temp_lengths, temp_eff_rad, temp_eff_length

#########
# Now we implement the alternative, Teeffellen model that tracks cell biomass and models its increase in lockstep with
# cell surface area. Note, however, that we are taking the converse situation - biomass increasing in proportion to new
# surface area, rather than the other way around.

# These functions, sim_growth_biomass_strain_stiffening and eq_sys_biomass_strain_stiffening appear to be correct,
# but so far my simulations are not giving a stable cell radius, even when seeded with values that supposedly solve the
# analytics for the biomass accumulation case. It's possible that this is to do with the specific value of biomass in
# each case, but that's not certain yet.


def sim_growth_biomass_strain_stiffening(temp_tvec, temp_ri, temp_li, temp_mi, temp_params):
    # This function takes the initial lengths and radii and simulates growth according to the abrupt, "all in one" model
    # of cell wall hydrolysis. However, it specifically makes sure that the cell balances its wall stresses prior to
    # inserting a new layer
    # As parameters, this takes the time vector, the rest initial radii, the rest initial lengths and the additional
    # parameters
    # time between each step of layer removal and insertion, in minutes. Usually set to 1.
    temp_dr = temp_params['dr']
    # Spacing between each layer.
    temp_eff_rad, temp_eff_length, temp_eff_mass = np.empty(temp_tvec.shape), np.empty(temp_tvec.shape), \
                                                   np.empty(temp_tvec.shape)
    temp_eff_rad[0], temp_eff_length[0], temp_eff_mass[0] = temp_ri[0], temp_li[0], temp_mi.copy()
    # Rest lengths at each timepoint.
    # Now we perform the iteration
    temp_radii, temp_lengths = [temp_ri.copy()], [temp_li.copy()]
    for ind in range(len(temp_tvec)-1):
        # if ind == 0:
        #     rint = temp_radii[-1].copy()  # initially we just equilibrate
        #     lint = temp_lengths[-1].copy()
        # else:
        # Let's now remove a layer
        rint = temp_radii[-1][:-1].copy()
        lint = temp_lengths[-1][:-1].copy()

        rprev = rint[0].copy()
        lprev = lint[0].copy()
        # Now we equilibriate to find the innermost radius
        root = fsolve(eq_sys_biomass_strain_stiffening, [rprev, lprev], args=tuple([rint, lint, temp_eff_mass[ind], temp_params]))
        # Now we add a new layer inside the innermost radius
        rint = np.concatenate(
            (np.array([root[0] - temp_dr]), rint))  # new layer inside new rest radius
        lint = np.concatenate(
            (np.array([root[1]]), lint))  # new layer has the new rest length.

        # Note that as soon as we add this layer, the minimum energy configuration will change slightly. However, that
        # doesn't really matter since the rest lengths are still as defined, and we determine the next iteration of
        # layer addition based on the equilibrium state at the time.
        temp_radii.append(rint.copy())
        temp_lengths.append(lint.copy())
        temp_eff_rad[ind+1], temp_eff_length[ind+1] = root[0] - temp_dr, root[1]
        temp_eff_mass[ind + 1] = temp_eff_mass[ind] + temp_params['k_rate'] * 2 * math.pi * \
                                 (temp_eff_rad[ind+1]*temp_eff_length[ind+1]-temp_eff_rad[ind]*temp_eff_length[ind])
    return temp_radii, temp_lengths, temp_eff_rad, temp_eff_length, temp_eff_mass


def eq_sys_biomass_strain_stiffening(temp_vals, temp_ri, temp_li, temp_mi, temp_params):
    # this function takes a vector of rest lengths and rest radii and calculates the strain in each case
    # based on inputs for the overall cell inner radius and length, temp_rval and temp_lval.
    # Here temp_vals is the proposed internal radius and cell length, while temp_ri and temp_li
    # are the rest radius and rest length for each layer.
    temp_rval = temp_vals[0]
    temp_lval = temp_vals[1]

    temp_dr = temp_params['dr']
    temp_alpha1 = temp_params['alpha1']
    temp_alpha2 = temp_params['alpha2']
    temp_strain_cutoff = temp_params['strain_cutoff']
    temp_nu = temp_params['nu']
    temp_PK_rat = temp_params['PK_rat']

    # We now calculate the radial strain = (r1-r0)/r0 and longitudinal strain = (l1-l0)/l0
    eps_r = (temp_rval + temp_dr * np.arange(0, len(temp_ri)).astype(
        float) - temp_ri) / temp_ri  # radial strain in each layer
    # based on equal spacing between layers of dr.
    eps_l = (temp_lval - temp_li) / temp_li  # longitudinal strain in each layer

    temp_alphas = temp_alpha1 * (eps_r <= temp_strain_cutoff) + temp_alpha2 * (
                eps_r > temp_strain_cutoff)  # this gives the
    # specific values for alpha for each layer in this configuration. Note that this is a vector.

    # Now we calculate the tensions in each layer
    sig_l = (temp_dr / (temp_alphas - temp_nu ** 2)) * (temp_alphas * eps_l + temp_nu * eps_r)
    # the longitudinal tension in each layer
    sig_r = (temp_dr / (temp_alphas - temp_nu ** 2)) * (temp_nu * eps_l + eps_r)
    # the radial tension in each layer
    # note the tension is normalized by the spacing of each layer. We assume each layer to have thickness dr.
    # Note that we now make P~m/V, with the proportionality constant absorbed into PK_rat.
    temp_out1 = np.sum(sig_r) - temp_PK_rat * np.absolute(temp_rval) * temp_mi / (math.pi * temp_rval**2 * temp_lval)
    # note that we use absolute to avoid spurious
    # negative solutions, since radius should always be positive.
    # this is the first constraint equation (this should equal 0)
    temp_out2 = np.sum((np.absolute(temp_rval) + temp_dr * np.arange(0, len(temp_ri)).astype(float)) * sig_l) \
                - temp_PK_rat * (np.absolute(temp_rval) ** 2) * temp_mi / (2 * math.pi * temp_rval**2 * temp_lval)
    # Note that we now make P~m/V
    # This is the second constraint equation which should also equal zero.
    return [temp_out1, temp_out2]

 ########################################
 # Still to be completed


def sim_growth_v1(temp_tvec, temp_ri, temp_li, temp_si, temp_params):
    # This function performs gradual hydrolysis, incorporating radial dependency of insertion rate or initial cell wall
    # stiffness on curvature of the cell wall. It is not yet finished as of 1/30/23.
    temp_radii, temp_lengths, temp_stiffnesses = [temp_ri.copy()], [temp_li.copy()], [temp_si.copy()]
    insertion_times = []  # vector of timepoints at which new layers are inserted
    temp_eff_rad, temp_eff_length = np.empty(temp_tvec.shape), np.empty(temp_tvec.shape)
    # vector of equilibrium radii at each timepoint
    numerical_error = 0.000000001

    temp_hydr_rate = temp_params['hydr_rate']
    temp_dt = temp_params['dt']
    temp_dr = temp_params['dr']
    temp_synth_rate = temp_params['synth_rate']

    for ind in np.arange(0, len(temp_tvec)):
        # Let's now remove a layer, add an inside layer at the previous equilibrium radius,
        # and then equilibriate again.
        rint = temp_radii[-1].copy()  # This is the basic state, where effectively nothing happens.
        lint = temp_lengths[-1].copy()
        sint = temp_stiffnesses[-1].copy()
        insertion = True  # dummy variable for now
        if ind == 0:
            # for the very first timestep, all we do is equilibrate and figure out when we are next inserting
            # a layer
            rprev = temp_radii[-1][0].copy()
            lprev = temp_lengths[-1][0].copy()
            root = fsolve(eq_sys_grad_hydr, [rprev, lprev], args=tuple([rint, lint, sint, temp_params]))

        else:
            sint = sint - temp_dt * temp_hydr_rate
            while sint[-1] < numerical_error:  # if one layer now has effectively zero (or even negative) stiffness
                sint = sint[:-1]  # in this case we remove the outer layer. Note that this should be robust, even if
                # multiple layers get removed simultaneously.
                rint = rint[:-1]
                lint = lint[:-1]

            insertion = -0.5 < (temp_tvec[ind] - insertion_times[-1]) / temp_dt <= 0.5
            # if an insertion should have taken place within this window
            if insertion:
                # insert a new layer at a radius equal to the previous equilibrium radius, at the previous eq.
                # length, with full stiffness.
                rint = np.concatenate((np.array([root[0] - temp_dr]), rint.copy()))
                lint = np.concatenate((np.array([root[1]]), lint.copy()))
                sint = np.concatenate((np.array([1.0]), sint.copy()))  # This part needs to change dependent on radius

            #         print(sint)
            # Now we update the equilibrium radius and length
            rprev = temp_radii[-1][0].copy()
            lprev = temp_lengths[-1][0].copy()
            root = fsolve(eq_sys_grad_hydr, [rprev, lprev], args=tuple([rint, lint, sint, temp_params]))

        if insertion:
            insertion_times.append(temp_tvec[ind] + 1.0 / insertion_rate(root[0], opt_rate=temp_synth_rate))
        temp_eff_rad[ind], temp_eff_length[ind] = root[0], root[1]
        # Updating the stored radii, lengths and stiffnesses
        temp_radii.append(rint.copy())
        temp_lengths.append(lint.copy())
        temp_stiffnesses.append(sint.copy())
    return temp_radii, temp_lengths, temp_stiffnesses, temp_eff_rad, temp_eff_length