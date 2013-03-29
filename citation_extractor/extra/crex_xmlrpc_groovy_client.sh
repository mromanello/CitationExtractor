#!/usr/bin/env groovy

import groovy.net.xmlrpc.*

def server = new XMLRPCServerProxy("http://137.73.122.221:8000/rpc/crex")

def result = server.version() + server.test()

println result
