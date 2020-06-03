#!groovy

@Library('katsdpjenkins') _
katsdp.killOldJobs()
katsdp.setDependencies(['ska-sa/katsdpdockerbase/master'])
katsdp.standardBuild()
katsdp.mail('sdp-dev+katsdpmodels@ska.ac.za')
