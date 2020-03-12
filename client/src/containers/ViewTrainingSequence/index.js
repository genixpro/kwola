import React, { Component } from 'react';
import {connect, Provider} from 'react-redux';
import { createStore, combineReducers } from 'redux';
import { reducer as reduxFormReducer } from 'redux-form';
import LayoutWrapper from '../../components/utility/layoutWrapper';
import PageTitle from '../../components/utility/paperTitle';
import Papersheet, { DemoWrapper } from '../../components/utility/papersheet';
import { FullColumn , HalfColumn, OneThirdColumn, TwoThirdColumn, Row, Column} from '../../components/utility/rowColumn';
import {withStyles} from "@material-ui/core";
import action from "../../redux/ViewTrainingSequence/actions";
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

class ViewTrainingSequence extends Component {
    state = {
        result: '',
    };

    componentDidMount() {
        store.dispatch(action.requestTrainingSequence(this.props.match.params.id));
    }


    getConfigurationForHeaderChart() {
        const trainingSequence = this.props.trainingSequence;
        const dataPoints = _.map(trainingSequence.trainingSteps, (step) => step.averageLoss);

        return {
            labels: ["January", "February", "March", "April", "May", "June", "July"],
            datasets: [
                {
                    label: "Average Training Loss",
                    fill: false,
                    lineTension: 0.1,
                    backgroundColor: "rgba(72,166,242,0.6)",
                    borderColor: "rgba(72,166,242,1)",
                    borderCapStyle: "butt",
                    borderDash: [],
                    borderDashOffset: 0.0,
                    borderJoinStyle: "miter",
                    pointBorderColor: "rgba(72,166,242,1)",
                    pointBackgroundColor: "#fff",
                    pointBorderWidth: 1,
                    pointHoverRadius: 5,
                    pointHoverBackgroundColor: "rgba(72,166,242,1)",
                    pointHoverBorderColor: "rgba(72,166,242,1)",
                    pointHoverBorderWidth: 2,
                    pointRadius: 1,
                    pointHitRadius: 10,
                    data: dataPoints
                }
            ]
        };
    }


    render() {
        const { result } = this.state;

        return (
            this.props.trainingSequence ?
                <LayoutWrapper>
                    <FullColumn>
                        <Row>
                            <HalfColumn>
                                <Papersheet>
                                    <Line data={this.getConfigurationForHeaderChart()} />
                                </Papersheet>
                            </HalfColumn>

                            <HalfColumn>
                                <Papersheet
                                    title={`Training Sequence ${this.props.trainingSequence._id.$oid}`}
                                    // subtitle={}
                                >
                                    <span>Status: {this.props.trainingSequence.status}<br/></span>

                                    <span>Start Time: {moment(this.props.trainingSequence.startTime).format('MMM Do, YYYY')}<br/></span>

                                    {
                                        this.props.trainingSequence.endTime ?
                                            <span>End Time: {moment(this.props.trainingSequence.endTime).format('MMM Do, YYYY')}<br/></span>
                                            : <span>End Time: N/A<br/></span>
                                    }
                                </Papersheet>
                            </HalfColumn>
                        </Row>

                        <Row>
                            <HalfColumn>
                                <Papersheet title={"Testing Sequences"}>
                                    <Table>
                                        <TableHead>
                                            <TableRow>
                                                <TableCell>Test Start Time</TableCell>
                                                <TableCell>Status</TableCell>
                                                <TableCell>Average Reward</TableCell>
                                            </TableRow>
                                        </TableHead>
                                        <TableBody>
                                            {(this.props.testingSequences || []).map(testingSequence => {
                                                return (
                                                    <TableRow key={testingSequence._id.$oid} hover={true} onClick={() => this.props.history.push(`/dashboard/testing_sequences/${testingSequence._id.$oid}`)}>
                                                        <TableCell>{testingSequence.startTime ? moment(new Date(testingSequence.startTime.$date)).format('HH:mm MMM Do') : null}</TableCell>
                                                        <TableCell>{testingSequence.status}</TableCell>
                                                        <TableCell>{testingSequence.averageReward}</TableCell>
                                                    </TableRow>
                                                );
                                            })}
                                        </TableBody>
                                    </Table>
                                </Papersheet>
                            </HalfColumn>
                            <HalfColumn>
                                <Papersheet title={"Training Steps"}>
                                    <Table>
                                        <TableHead>
                                            <TableRow>
                                                <TableCell>Status</TableCell>
                                                <TableCell>Step Start Time</TableCell>
                                                <TableCell>Average Loss</TableCell>
                                            </TableRow>
                                        </TableHead>
                                        <TableBody>
                                            {(this.props.trainingSteps || []).map(trainingStep => {
                                                return (
                                                    <TableRow key={trainingStep._id.$oid} hover={true} onClick={() => this.props.history.push(`/dashboard/training_steps/${trainingStep._id.$oid}`)}>
                                                        <TableCell>{trainingStep.startTime ? moment(new Date(trainingStep.startTime.$date)).format('HH:mm MMM Do') : null}</TableCell>
                                                        <TableCell>{trainingStep.status}</TableCell>
                                                        <TableCell>{trainingStep.averageLoss}</TableCell>
                                                    </TableRow>
                                                );
                                            })}
                                        </TableBody>
                                    </Table>


                                </Papersheet>
                            </HalfColumn>
                        </Row>

                    </FullColumn>
                </LayoutWrapper>
                : null

        );
    }
}

const mapStateToProps = (state) => {return { ...state.ViewTrainingSequence} };
export default connect(mapStateToProps)(ViewTrainingSequence);

