<?xml version="1.0" encoding="UTF-8"?>
<xsl:stylesheet xmlns:xsl="http://www.w3.org/1999/XSL/Transform" xmlns:xs="http://www.w3.org/2001/XMLSchema" exclude-result-prefixes="xs" xmlns:xd="http://www.oxygenxml.com/ns/doc/xsl" version="1.0">
    <xd:doc scope="stylesheet">
        <xd:desc>
            <xd:p><xd:b>Created on:</xd:b> Oct 17, 2010</xd:p>
            <xd:p><xd:b>Author:</xd:b> 56k</xd:p>
            <xd:p/>
        </xd:desc>
    </xd:doc>
    <xsl:output indent="yes" encoding="UTF-8" method="html"/>
    <xsl:template match="/">
        <xsl:comment> output of <xsl:value-of select="reply/@service"/> version <xsl:value-of select="reply/@version"/>
        </xsl:comment>
        <div>
            <xsl:apply-templates/>
        </div>
    </xsl:template>
    <xsl:template match="instance">
        <div class="instance">
            <xsl:apply-templates select="token"/>
        </div>
        <div id="scores">
            <xsl:apply-templates select="//token/tags"></xsl:apply-templates>
        </div>        
    </xsl:template>
    <xsl:template match="token">
        <span>
            <xsl:attribute name="class">
                token_<xsl:value-of select="@label"/>
            </xsl:attribute>
            <xsl:attribute name="id">
                <xsl:value-of select="@id"/>
            </xsl:attribute>
            <xsl:value-of select="concat(@value,'  ')"/>
        </span>
    </xsl:template>
    <xsl:template match="tags">
        <div>
            <xsl:attribute name="class">score</xsl:attribute>
            <xsl:attribute name="id">tok-<xsl:value-of select="../@id"/></xsl:attribute>
            <ul>
                <li>token: <xsl:value-of select="../@value"/></li>
                <li>features: <xsl:value-of select="../features/text()"/></li>
                <xsl:for-each select="tag">
                    <li><xsl:value-of select="concat(@label,': ',@prob)"/></li>
                </xsl:for-each>
            </ul>
        </div>
    </xsl:template>
</xsl:stylesheet>
