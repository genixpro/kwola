import React, { Component } from 'react';
import {connect, Provider} from 'react-redux';
import { createStore, combineReducers } from 'redux';
import { reducer as reduxFormReducer } from 'redux-form';
import LayoutWrapper from '../../components/utility/layoutWrapper';
import PageTitle from '../../components/utility/paperTitle';
import Papersheet, { DemoWrapper } from '../../components/utility/papersheet';
import { FullColumn , HalfColumn, OneThirdColumn, TwoThirdColumn, Row, Column} from '../../components/utility/rowColumn';
import {withStyles} from "@material-ui/core";
import action from "../../redux/ViewExecutionSession/actions";
import {store} from "../../redux/store";
import SingleCard from '../Shuffle/singleCard.js';
import BoxCard from '../../components/boxCard';
import Img7 from '../../images/7.jpg';
import user from '../../images/user.jpg';
import ActionButton from "../../components/mail/singleMail/actionButton";
import {Button} from "../UiElements/Button/button.style";
import Icon from "../../components/uielements/icon";
import moment from 'moment';
import _ from "underscore";
import {Chip, Wrapper} from "../UiElements/Chips/chips.style";
import Avatar from "../../components/uielements/avatars";
import {Table} from "../ListApplications/materialUiTables.style";
import {TableBody, TableCell, TableHead, TableRow} from "../../components/uielements/table";
import { Line } from "react-chartjs-2";

class ViewExecutionSession extends Component {
    state = {
        result: '',
    };

    componentDidMount() {
        store.dispatch(action.requestExecutionSession(this.props.match.params.id));
    }


    render() {
        const { result } = this.state;

        return (
            this.props.executionSession ?
                <LayoutWrapper>
                    <FullColumn>
                        <Row>
                            <HalfColumn>
                                <Papersheet>
                                    <video controls style={{"width": "100%"}}>
                                        <source src={`http://localhost:8000/api/execution_sessions/${this.props.executionSession._id.$oid}/video`} type="video/mp4" />
                                        <span>Your browser does not support the video tag.</span>
                                    </video>
                                </Papersheet>
                            </HalfColumn>

                            <HalfColumn>
                                <Papersheet
                                    title={`Execution Session ${this.props.executionSession._id.$oid}`}
                                    // subtitle={}
                                >

                                    <span>Start Time: {moment(this.props.executionSession.startTime).format('MMM Do, YYYY')}<br/></span>

                                    {
                                        this.props.executionSession.endTime ?
                                            <span>End Time: {moment(this.props.executionSession.endTime).format('MMM Do, YYYY')}<br/></span>
                                            : <span>End Time: N/A<br/></span>
                                    }
                                </Papersheet>
                            </HalfColumn>
                        </Row>

                        <Row>
                            <FullColumn>
                                <Papersheet title={"Execution Traces"}>
                                    <Table>
                                        <TableHead>
                                            <TableRow>
                                                <TableCell>Frame</TableCell>
                                                <TableCell>Action Type</TableCell>
                                                <TableCell>Action X</TableCell>
                                                <TableCell>Action Y</TableCell>
                                                <TableCell>Cumulative Branch Coverage</TableCell>
                                                <TableCell>New Branch Executed</TableCell>
                                            </TableRow>
                                        </TableHead>
                                        <TableBody>
                                            {(this.props.executionTraces || []).map(trace => {
                                                return (
                                                    <TableRow key={trace._id.$oid} hover={true} onClick={() => this.props.history.push(`/dashboard/execution_traces/${trace._id.$oid}`)} >
                                                        <TableCell>{trace.frameNumber + 1}</TableCell>
                                                        <TableCell>{trace.actionPerformed.x.toString()}</TableCell>
                                                        <TableCell>{trace.actionPerformed.y.toString()}</TableCell>
                                                        <TableCell>{trace.actionPerformed.type.toString()}</TableCell>
                                                        <TableCell>{(trace.cumulativeBranchCoverage * 100).toFixed(3)}%</TableCell>
                                                        <TableCell>{trace.didNewBranchesExecute.toString()}</TableCell>
                                                    </TableRow>
                                                );
                                            })}
                                        </TableBody>
                                    </Table>
                                </Papersheet>
                            </FullColumn>
                        </Row>

                    </FullColumn>
                </LayoutWrapper>
                : null

        );
    }
}

const mapStateToProps = (state) => {return { ...state.ViewExecutionSession} };
export default connect(mapStateToProps)(ViewExecutionSession);

