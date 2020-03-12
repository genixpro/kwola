import React, { Component } from 'react';
import {connect, Provider} from 'react-redux';
import { createStore, combineReducers } from 'redux';
import { reducer as reduxFormReducer } from 'redux-form';
import LayoutWrapper from '../../components/utility/layoutWrapper';
import PageTitle from '../../components/utility/paperTitle';
import Papersheet, { DemoWrapper } from '../../components/utility/papersheet';
import { FullColumn , HalfColumn, OneThirdColumn, TwoThirdColumn, Row, Column} from '../../components/utility/rowColumn';
import {withStyles} from "@material-ui/core";
import action from "../../redux/ViewExecutionTrace/actions";
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

class ViewExecutionTrace extends Component {
    state = {
        result: '',
    };

    componentDidMount() {
        store.dispatch(action.requestExecutionTrace(this.props.match.params.id));
    }


    render() {
        const { result } = this.state;

        return (
            this.props.executionTrace ?
                <LayoutWrapper>
                    <FullColumn>
                        <Row>
                            <FullColumn>
                                <Papersheet
                                    title={`Execution Trace ${this.props.executionTrace._id.$oid}`}
                                    // subtitle={}
                                >

                                    <pre>
                                        {JSON.stringify(this.props.executionTrace, null, 4)}
                                    </pre>

                                </Papersheet>
                            </FullColumn>
                        </Row>
                    </FullColumn>
                </LayoutWrapper>
                : null

        );
    }
}

const mapStateToProps = (state) => {return { ...state.ViewExecutionTrace} };
export default connect(mapStateToProps)(ViewExecutionTrace);

