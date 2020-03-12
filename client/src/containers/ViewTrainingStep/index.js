import React, { Component } from 'react';
import {connect, Provider} from 'react-redux';
import { createStore, combineReducers } from 'redux';
import { reducer as reduxFormReducer } from 'redux-form';
import LayoutWrapper from '../../components/utility/layoutWrapper';
import PageTitle from '../../components/utility/paperTitle';
import Papersheet, { DemoWrapper } from '../../components/utility/papersheet';
import { FullColumn , HalfColumn, OneThirdColumn, TwoThirdColumn, Row, Column} from '../../components/utility/rowColumn';
import {withStyles} from "@material-ui/core";
import action from "../../redux/ViewTrainingStep/actions";
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

class ViewTrainingStep extends Component {
    state = {
        result: '',
    };

    componentDidMount() {
        store.dispatch(action.requestTrainingStep(this.props.match.params.id));
    }


    render() {
        const { result } = this.state;

        return (
            this.props.trainingStep ?
                <LayoutWrapper>
                    <FullColumn>
                        <Row>
                            <HalfColumn>
                                <Papersheet>

                                </Papersheet>
                            </HalfColumn>

                            <HalfColumn>
                                <Papersheet
                                    title={`Training Step ${this.props.trainingStep._id.$oid}`}
                                    // subtitle={}
                                >
                                    <span>Status: {this.props.trainingStep.status}<br/></span>

                                    <span>Start Time: {moment(this.props.trainingStep.startTime).format('MMM Do, YYYY')}<br/></span>

                                    {
                                        this.props.trainingStep.endTime ?
                                            <span>End Time: {moment(this.props.trainingStep.endTime).format('MMM Do, YYYY')}<br/></span>
                                            : <span>End Time: N/A<br/></span>
                                    }
                                </Papersheet>
                            </HalfColumn>
                        </Row>

                        <Row>
                            <FullColumn>
                                <Papersheet title={"Execution Sessions"}>
                                    <Table>
                                        <TableHead>
                                            <TableRow>
                                                <TableCell>Test Start Time</TableCell>
                                                <TableCell>Total Reward</TableCell>
                                            </TableRow>
                                        </TableHead>
                                        <TableBody>
                                            {(this.props.executionSessions || []).map(executionSession => {
                                                return (
                                                    <TableRow key={executionSession._id.$oid} hover={true} onClick={() => this.props.history.push(`/dashboard/execution_sessions/${executionSession._id.$oid}`)} >
                                                        <TableCell>{executionSession.startTime ? moment(new Date(executionSession.startTime.$date)).format('HH:mm MMM Do') : null}</TableCell>
                                                        <TableCell>{executionSession.totalReward}</TableCell>
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

const mapStateToProps = (state) => {return { ...state.ViewTrainingStep} };
export default connect(mapStateToProps)(ViewTrainingStep);

